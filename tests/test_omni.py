"""Image, video and audio input on a Nemotron Omni base, from a tiny random checkpoint on disk.

The checkpoint is written as the released ones are: the Omni wrapper's legacy config.json
(model_type NemotronH_Nano_Omni_Reasoning_V3, `llm_config` / `vision_config` /
`sound_config` with the parakeet model_type that transformers' config rejects, video
pruning on, `auto_map` to Hub code that does not exist) and
the legacy weight names (`language_model.backbone.*` with one tensor per routed expert,
`vision_model.radio_model.*` with fused qkv, `mlp1.*`, `sound_encoder.encoder.*`,
`sound_projection.*`). transformers' converter and omni.py's config fixes must both work for
the model to load with the weights it was saved with.

The Mamba kernels take CUDA tensors only, so on a CUDA host with mamba-ssm everything runs
on the GPU; elsewhere on the PyTorch fallback.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "NemotronH_Omni_Reasoning_V3"):
    pytest.skip("this transformers has no nemotron_h_omni", allow_module_level=True)
av = pytest.importorskip("av")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from test_ddp import _examples  # noqa: E402
from test_nemotron_train import _llm_config  # noqa: E402

from strands_decider import omni as O  # noqa: E402
from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    StrandsDeciderConfig,
    StrandsDeciderModel,
    apply_temperature,
    gather_options,
    masked_log_softmax,
)
from strands_decider.prompting import render_question  # noqa: E402
from strands_decider.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "shared_experts.up_proj",
    "shared_experts.down_proj",
]
# Media tokens at the released tokenizer's ids: <image> 18, <img> 19, </img> 20,
# <so_embedding> 27, <so_start> 28, <so_end> 29.
SPECIAL = [
    "<pad>",
    "<eos>",
    "<bos>",
    *[f"<r{i}>" for i in range(3, 18)],
    "<image>",
    "<img>",
    "</img>",
    *[f"<r{i}>" for i in range(21, 27)],
    "<so_embedding>",
    "<so_start>",
    "<so_end>",
]


def _tokenizer(path, rows):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    from strands_decider.prompting import build_prompt

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(
        vocab_size=500, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator([build_prompt(ex.state, ex.to_question())[0] for ex in rows], trainer)
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>"
    )
    for t, i in (
        ("<image>", 18),
        ("<img>", 19),
        ("</img>", 20),
        ("<so_embedding>", 27),
        ("<so_start>", 28),
    ):
        assert fast.convert_tokens_to_ids(t) == i
    fast.save_pretrained(path)
    return fast


def _legacy_config(vocab: int) -> dict:
    llm = _llm_config(vocab)  # no num_nextn_predict_layers, as in the released config
    return {
        "architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
        "auto_map": {"AutoConfig": "configuration.Nope", "AutoModel": "modeling.Nope"},
        "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
        "downsample_ratio": 0.5,
        "img_context_token_id": 18,
        "sound_context_token_id": 27,
        "vit_hidden_size": 32,
        "projector_hidden_size": 64,
        "norm_mean": MEAN,
        "norm_std": STD,
        "video_pruning_rate": 0.7,
        "torch_dtype": "float32",
        "llm_config": llm,
        "vision_config": {
            "architectures": ["RADIOModel"],
            "auto_map": {"AutoConfig": "nvidia/C-RADIOv4-H--hf_model.RADIOConfig"},
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 16,
            "max_img_size": 2048,
            "video_temporal_patch_size": 2,
        },
        "sound_config": {
            "model_type": "parakeet",  # the field transformers' config rejects
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "conv_kernel_size": 9,
            "subsampling_conv_channels": 8,
            "subsampling_factor": 8,
            "num_mel_bins": 128,
            "projection_hidden_size": 32,
            "projection_bias": False,
            "convolution_bias": False,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "sampling_rate": 16000,
        },
    }


def _to_legacy(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Native parameter names back to the released checkpoint's."""
    out: dict[str, torch.Tensor] = {}
    qkv: dict[str, dict[str, torch.Tensor]] = {}
    pg = "vision_model.radio_model.model.patch_generator."
    for k, v in state.items():
        v = v.detach().clone().contiguous()
        if k.startswith("language_model.model."):
            name = "language_model.backbone." + k.removeprefix("language_model.model.")
            if name.endswith((".experts.up_proj", ".experts.down_proj")):
                stem, proj = name.rsplit(".", 1)
                for e in range(v.shape[0]):
                    out[f"{stem}.{e}.{proj}.weight"] = v[e].clone()
                continue
            out[name] = v
        elif k.startswith("language_model."):
            out[k] = v  # lm_head
        elif k == "vision_model.summary_idxs":
            continue  # the released checkpoint has none
        elif k.startswith("vision_model.embeddings."):
            rest = k.removeprefix("vision_model.embeddings.")
            for a, b in (
                ("video_patch_projection", "video_embedder"),
                ("patch_projection", "embedder"),
                ("position_embedding", "pos_embed"),
                ("cls_register_token", "cls_token.token"),
            ):
                if rest.startswith(a):
                    out[pg + b + rest[len(a) :]] = v
                    break
            else:
                raise KeyError(k)
        elif k.startswith("vision_model.encoder.layer."):
            rest = k.removeprefix("vision_model.encoder.layer.")
            layer, sub = rest.split(".", 1)
            base = f"vision_model.radio_model.model.blocks.{layer}."
            if sub.startswith(("attention.q_proj", "attention.k_proj", "attention.v_proj")):
                proj, kind = sub.split(".")[1], sub.split(".")[-1]
                qkv.setdefault(base + "attn.qkv." + kind, {})[proj] = v
            elif sub.startswith("attention.o_proj"):
                out[base + "attn.proj" + sub[len("attention.o_proj") :]] = v
            else:
                out[base + sub] = v
        elif k.startswith("vision_model."):
            out["vision_model.radio_model." + k.removeprefix("vision_model.")] = v
        elif k.startswith("multi_modal_projector."):
            rest = k.removeprefix("multi_modal_projector.")
            for a, b in (("layer_norm", "mlp1.0"), ("linear_1", "mlp1.1"), ("linear_2", "mlp1.3")):
                if rest.startswith(a):
                    out[b + rest[len(a) :]] = v
        elif k.startswith("audio_tower."):
            out["sound_encoder.encoder." + k.removeprefix("audio_tower.")] = v
        elif k.startswith("embed_audio."):
            rest = k.removeprefix("embed_audio.")
            for a, b in (("layer_norm", "norm"), ("linear_1", "linear1"), ("linear_2", "linear2")):
                if rest.startswith(a):
                    out["sound_projection." + b + rest[len(a) :]] = v
        else:
            raise KeyError(k)
    for name, parts in qkv.items():
        out[name] = torch.cat([parts["q_proj"], parts["k_proj"], parts["v_proj"]])
    out["vision_model.radio_model.input_conditioner.norm_mean"] = torch.tensor(MEAN).view(3, 1, 1)
    out["vision_model.radio_model.input_conditioner.norm_std"] = torch.tensor(STD).view(3, 1, 1)
    out["sound_encoder.encoder.feature_extractor.featurizer.fb"] = torch.rand(1, 128, 257)
    return out


@pytest.fixture(scope="module")
def omni_base(tmp_path_factory):
    root = tmp_path_factory.mktemp("omni")
    base = root / "base"
    rows = [ex for ex in _examples(80, seed=1) if ex.n_options <= 9]
    tok = _tokenizer(base, rows)
    cfg_json = _legacy_config(len(tok))
    (base / "config.json").write_text(json.dumps(cfg_json))
    (base / "preprocessor_config.json").write_text(json.dumps({"norm_mean": MEAN, "norm_std": STD}))
    torch.manual_seed(0)
    native = transformers.NemotronH_Omni_Reasoning_V3(O.omni_config(str(base), None)).eval()
    state = {k: v for k, v in native.state_dict().items()}
    save_file(_to_legacy(state), str(base / "model.safetensors"))
    # A checkpoint as text training saves one: LoRA on the text decoder and a pointer head,
    # random but non-zero so they change the answers.
    config = StrandsDeciderConfig(
        base_model=str(base),
        torch_dtype="float32",
        lora_targets=TARGETS,
        pointer_dim=16,
        max_length=2048,
    )
    model = StrandsDeciderModel.from_pretrained_base(config)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n or n.startswith("head.norm"):
                p.normal_(std=0.3)
    ckpt = root / "ckpt"
    model.save_pretrained(str(ckpt))
    return base, ckpt, state, native


@pytest.fixture(scope="module")
def engines(omni_base):
    _, ckpt, _, _ = omni_base
    omni = O.OmniEngine(O.OmniDeciderModel.load(str(ckpt)), O.OmniEngineConfig(device=DEVICE))
    text = SystemOneEngine(StrandsDeciderModel.load(str(ckpt)), EngineConfig(device=DEVICE))
    return omni, text


def _png(w=64, h=48, seed=0) -> str:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mp4(frames=4, fps=2, w=64, h=48) -> str:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        s = out.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(frames):
            img = np.full((h, w, 3), (i * 40) % 256, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return base64.b64encode(buf.getvalue()).decode()


def _wav(seconds=1.0) -> str:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="wav") as out:
        s = out.add_stream("pcm_s16le", rate=16000, layout="mono")
        t = np.arange(int(seconds * 16000)) / 16000
        pcm = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = 16000
        for pkt in s.encode(frame):
            out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return base64.b64encode(buf.getvalue()).decode()


QUESTIONS = {
    "busy": NoulQuestion(instructions="Is the scene busy?"),
    "kind": ChoiceQuestion(
        instructions="What is it?", criteria={"cat": None, "car": "a vehicle", "cup": None}
    ),
}


def test_the_omni_base_loads_with_its_own_weights_and_frozen_encoders(omni_base, engines):
    _, _, state, _ = omni_base
    omni, _ = engines
    m = omni.model
    torso = m.torso.base_model.model.state_dict()  # parameters and buffers
    for k, v in state.items():
        if k.startswith("language_model.model."):
            name = k.removeprefix("language_model.model.")
            head, _, leaf = name.rpartition(".")
            got = torso.get(
                name, torso.get(f"{head}.base_layer.{leaf}")
            )  # LoRA keeps it in base_layer
            assert got is not None and torch.equal(got.detach().cpu(), v), k
    enc = m.encoders.state_dict()
    for k, v in state.items():
        if (
            k.startswith(
                ("vision_model.", "multi_modal_projector.", "audio_tower.", "embed_audio.")
            )
            and k in enc
        ):
            assert torch.equal(enc[k].detach().cpu(), v), k
    assert not any(p.requires_grad for p in m.encoders.parameters())
    assert isinstance(m.encoders.language_model, torch.nn.Identity)
    lora = [n for n, _ in m.torso.named_modules() if n.endswith("lora_A")]
    assert lora and all(n.startswith("base_model.model.layers.") for n in lora)
    assert {x.time_step_limit for x in m.torso.modules() if hasattr(x, "time_step_limit")} == {
        (0.0, 1e30)
    }


def test_text_answers_are_the_text_engines(engines):
    omni, text = engines
    req = SystemOneRequest(state="A quiet street with one parked car.", questions=QUESTIONS)
    a, b = omni.evaluate(req).answers, text.evaluate(req).answers
    for k in QUESTIONS:
        assert a[k].model_dump() == b[k].model_dump(), k


def _full_forward(engine: O.OmniEngine, state, media: O.Media, question) -> torch.Tensor:
    """One question through one uncached forward over state + media + question."""
    rq = render_question(question)
    s, q, offsets = engine._fit_media(state, media, [rq.text])
    ids = torch.tensor([s + q[0]], device=engine.device)
    with torch.inference_mode():
        emb = engine.media_embeds(ids, media)
        hidden = engine.model.torso(
            inputs_embeds=emb, attention_mask=torch.ones_like(ids), return_dict=True
        ).last_hidden_state
        pooled = hidden[:, -1].to(torch.float32)
        idx = engine._option_idx([rq], 0, offsets) + len(s)
        options = gather_options(hidden, idx).to(torch.float32)
        logits = apply_temperature(
            engine.model.head(pooled, options), engine._temperatures([rq.kind])
        )
        return masked_log_softmax(logits, torch.tensor([rq.n_slots], device=engine.device)).exp()[
            0, : rq.n_slots
        ]


@pytest.mark.parametrize(
    "media",
    [
        {"images": [_png()]},
        {"images": [_png(seed=1), _png(80, 40, seed=2)]},
        {"videos": [_mp4()]},
        {"audio": [_wav()]},
        {"images": [_png()], "audio": [_wav(0.5)]},
        {"videos": [_mp4()], "audio": [_wav(0.5)]},
    ],
    ids=["image", "two-images", "video", "audio", "image+audio", "video+audio"],
)
def test_media_answers_through_the_cached_prefix_equal_a_full_forward(engines, media):
    omni, _ = engines
    req = SystemOneRequest(state="Look at this.", questions=QUESTIONS, **media)
    got = omni.evaluate(req).answers
    prepared = omni.prepare(req)
    for name, question in QUESTIONS.items():
        rq = render_question(question)
        full = _full_forward(omni, req.state, prepared, question).cpu().tolist()
        ans = got[name].model_dump()
        if rq.kind == "noul":
            assert ans["noul"] == pytest.approx(full[rq.slot_labels.index("true")], abs=1e-4), name
        else:
            for label, p in zip(rq.slot_labels, full, strict=True):
                assert ans["probabilities"][label] == pytest.approx(p, abs=1e-4), (name, label)
    # and the media are read: an empty state answers differently with and without them
    blind = omni.evaluate(SystemOneRequest(state="Look at this.", questions=QUESTIONS)).answers
    assert any(blind[k].model_dump() != got[k].model_dump() for k in QUESTIONS)


def test_media_tokens_in_the_text_are_refused(engines):
    omni, _ = engines
    with pytest.raises(ValueError, match="placeholder tokens"):
        omni.evaluate(
            SystemOneRequest(state="sneaky <image> here", questions=QUESTIONS, images=[_png()])
        )
    with pytest.raises(ValueError, match="placeholder tokens"):
        omni.evaluate(
            SystemOneRequest(
                state="ok",
                images=[_png()],
                questions={"q": NoulQuestion(instructions="Is <so_embedding> there?")},
            )
        )


def test_limits_and_the_window_are_enforced(engines):
    omni, _ = engines
    with pytest.raises(ValueError, match="at most 4"):
        omni.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, images=[_png()] * 5))
    with pytest.raises(ValueError, match="at most 1"):
        omni.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, videos=[_mp4()] * 2))
    small = omni.model.config.max_length
    try:
        omni.model.config.max_length = 200  # one image is at least 256 tokens
        with pytest.raises(ValueError, match="exceeds the context window"):
            omni.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, images=[_png()]))
    finally:
        omni.model.config.max_length = small


def test_the_text_engine_refuses_video_and_audio(engines):
    _, text = engines
    for media in ({"videos": [_mp4()]}, {"audio": [_wav()]}):
        with pytest.raises(ValueError, match="text only"):
            text.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, **media))


def test_the_config_fixes_are_applied(omni_base):
    base, _, _, _ = omni_base
    raw = json.loads((base / "config.json").read_text())
    d = O.omni_config_dict(raw)
    assert (
        "model_type" not in d["sound_config"]
        and "auto_map" not in d
        and "auto_map" not in d["vision_config"]
    )
    assert d["llm_config"]["num_nextn_predict_layers"] == 0 and d["video_pruning_rate"] == 0.0
    assert raw["sound_config"]["model_type"] == "parakeet"  # the input is left as it was


@pytest.mark.parametrize("combo", ["images+audio", "video+audio"])
def test_media_embeddings_are_the_native_models(omni_base, engines, combo):
    """Our splice (features into the placeholders, in prompt order) against transformers' own
    forward: the input embeddings its language model receives for the same request."""
    _, _, _, native = omni_base
    omni, _ = engines
    visual = (
        {"images": [_png(), _png(80, 40, seed=3)]}
        if combo == "images+audio"
        else {"videos": [_mp4()]}
    )
    req = SystemOneRequest(state="Look.", questions=QUESTIONS, audio=[_wav(0.7)], **visual)
    media = omni.prepare(req)
    s, _, _ = omni._fit_media(req.state, media, [render_question(QUESTIONS["busy"]).text])
    ids = torch.tensor([s], device=DEVICE)
    ours = omni.media_embeds(ids, media)
    native = native.to(DEVICE)
    seen = {}

    def grab(_module, _args, kwargs):
        seen["embeds"] = kwargs["inputs_embeds"].detach()

    if media.images:
        inputs = {
            "pixel_values": torch.cat([i.pixel_values for i in media.images]).to(DEVICE),
            "image_grid_hw": torch.tensor([i.grid_hw for i in media.images], device=DEVICE),
        }
    else:
        # transformers' forward torch.cat()s the video features, but get_video_features returns
        # them as one tensor; hand them in through its mm_encoder_outputs as the tuple it expects.
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        with torch.inference_mode():
            feats = native.get_video_features(media.videos[0].frames.to(DEVICE)).pooler_output
        inputs = {
            "mm_encoder_outputs": {"video": BaseModelOutputWithPooling(pooler_output=tuple(feats))}
        }
    hook = native.language_model.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with torch.inference_mode():
            native(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                input_features=media.audio[0].features.unsqueeze(0).to(DEVICE),
                input_features_mask=media.audio[0].mask.unsqueeze(0).to(DEVICE),
                **inputs,
            )
    finally:
        hook.remove()
        native.to("cpu")
    assert seen["embeds"].shape == ours.shape
    assert torch.allclose(seen["embeds"].float(), ours.float(), atol=1e-5)


def test_images_and_a_video_together_are_refused(engines):
    omni, _ = engines
    with pytest.raises(ValueError, match="images or a video, not both"):
        omni.evaluate(
            SystemOneRequest(state="x", questions=QUESTIONS, images=[_png()], videos=[_mp4()])
        )
