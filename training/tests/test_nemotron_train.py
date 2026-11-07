"""Training on a Nemotron-3-Nano-Omni torso, from a tiny random-weight checkpoint on disk.

The checkpoint is written in the Omni layout (`language_model.backbone.*`, one tensor per
routed expert, an untied `lm_head`, a vision weight, and an `auto_map` to Hub code that
does not exist, so a loader that ran it would fail). The real loader keeps the text decoder
through transformers' nemotron_h; then LoRA, gradient checkpointing, two optimizer steps of
`train()`, `load()`, and serving with and without the shared-prefix cache.

On a CUDA host with mamba-ssm installed the Mamba layers run its kernels, which take CUDA
tensors only, so everything runs on the GPU there; elsewhere on the PyTorch fallback.
"""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "NemotronHConfig"):
    pytest.skip("this transformers has no nemotron_h", allow_module_level=True)

from safetensors.torch import load_file, save_file  # noqa: E402
from test_ddp import _examples  # noqa: E402
from test_gemma4_train import _tokenizer  # noqa: E402

import strands_decider.modeling as M  # noqa: E402
import training.train as T  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from training.data.format import write_jsonl  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj"]
TARGETS += ["shared_experts.up_proj", "shared_experts.down_proj"]
LAYERS = ["linear_attention", "moe", "full_attention", "linear_attention", "moe"]
N_EXPERTS = 4


def _llm_config(vocab: int) -> dict:
    return dict(
        vocab_size=vocab,
        hidden_size=64,
        layers_block_type=LAYERS,
        num_hidden_layers=len(LAYERS),
        # 8 heads: causal-conv1d needs in_proj's width (2*64 + 2*16 + heads) a multiple of 8.
        mamba_num_heads=8,
        mamba_head_dim=8,
        n_groups=1,
        ssm_state_size=16,
        chunk_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=32,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        time_step_limit=[0.0, 1e30],  # the real checkpoint's; transformers ignores it
        # What the real llm_config carries; the loader must not follow it.
        architectures=["NemotronHForCausalLM"],
        auto_map={"AutoConfig": "configuration_nemotron_h.NemotronHConfig"},
    )


def _omni_checkpoint(path, vocab: int) -> dict[str, torch.Tensor]:
    """Save a tiny Omni-layout checkpoint; return the decoder weights by native name."""
    torch.manual_seed(0)
    llm = _llm_config(vocab)
    cfg = transformers.NemotronHConfig(
        **{k: v for k, v in llm.items() if k not in ("architectures", "auto_map")}
    )
    lm = transformers.NemotronHForCausalLM(cfg)
    native = {k: v.detach().clone().contiguous() for k, v in lm.state_dict().items()}
    disk = {}
    for name, t in native.items():
        if name == "lm_head.weight":
            disk["language_model.lm_head.weight"] = t
            continue
        on_disk = "language_model.backbone." + name.removeprefix("model.")
        if name.endswith((".experts.up_proj", ".experts.down_proj")):
            stem, proj = on_disk.rsplit(".", 1)  # ...mixer.experts, up_proj
            for e in range(t.shape[0]):  # the checkpoint keeps one tensor per routed expert
                disk[f"{stem}.{e}.{proj}.weight"] = t[e].clone()
        else:
            disk[on_disk] = t
    disk["vision_model.radio_model.patch.weight"] = torch.zeros(2, 2)
    save_file(disk, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
                "architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
                "auto_map": {
                    "AutoConfig": "configuration.NemotronH_Nano_Omni_Reasoning_V3_Config",
                    "AutoModel": "modeling.NemotronH_Nano_Omni_Reasoning_V3",
                },
                "llm_config": llm,
                "vision_config": {"model_type": "radio"},
                "torch_dtype": "float32",
            }
        )
    )
    return {k.removeprefix("model."): v for k, v in native.items() if k != "lm_head.weight"}


@pytest.fixture
def base(tmp_path):
    rows = [ex for ex in _examples(80, seed=1) if ex.n_options <= 9]
    write_jsonl(str(tmp_path / "train.jsonl"), rows)
    write_jsonl(str(tmp_path / "val.jsonl"), _examples(10, seed=2))
    path = tmp_path / "base"
    tok = _tokenizer(path, rows)
    weights = _omni_checkpoint(path, len(tok))
    return path, rows, weights


def _config(path, **kw) -> StrandsDeciderConfig:
    return StrandsDeciderConfig(
        base_model=str(path), torch_dtype="float32", lora_targets=TARGETS, **kw
    )


def test_the_decoder_loads_alone_with_the_checkpoint_weights(base):
    path, _, weights = base
    torso = StrandsDeciderModel._load_torso(_config(path), None, None)
    assert isinstance(torso, transformers.NemotronHModel)
    assert not hasattr(torso, "lm_head")
    state = torso.state_dict()
    assert set(state) == set(weights)
    for k, v in weights.items():
        assert torch.equal(state[k], v), k
    assert StrandsDeciderModel.is_hybrid(torso)
    # the Mamba kernels clamp the step size at the checkpoint's limit, not time_step_min
    limits = {m.time_step_limit for m in torso.modules() if hasattr(m, "time_step_limit")}
    assert limits == {(0.0, 1e30)}


def test_a_missing_decoder_weight_is_refused(base):
    path, _, _ = base
    disk = load_file(str(path / "model.safetensors"))
    disk.pop("language_model.backbone.layers.0.mixer.in_proj.weight")
    save_file(disk, str(path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="in_proj"):
        StrandsDeciderModel._load_torso(_config(path), None, None)


def test_lora_reaches_attention_mamba_in_proj_and_shared_experts_only(base):
    path, _, _ = base
    model = StrandsDeciderModel.from_pretrained_base(_config(path))
    adapted = {
        n.split(".layers.")[1].rsplit(".lora_A", 1)[0]
        for n, _ in model.torso.named_parameters()
        if "lora_A" in n
    }
    assert adapted == {
        "0.mixer.in_proj",
        "1.mixer.shared_experts.up_proj",
        "1.mixer.shared_experts.down_proj",
        "2.mixer.q_proj",
        "2.mixer.k_proj",
        "2.mixer.v_proj",
        "2.mixer.o_proj",
        "3.mixer.in_proj",
        "4.mixer.shared_experts.up_proj",
        "4.mixer.shared_experts.down_proj",
    }
    assert not any(p.requires_grad for n, p in model.torso.named_parameters() if ".experts." in n)


@pytest.mark.parametrize("target", ["out_proj", "conv1d"])
def test_lora_on_a_kernel_bypassed_mamba_module_is_refused(base, target):
    path, _, _ = base
    config = _config(path)
    config.lora_targets = [*TARGETS, target]
    with pytest.raises(ValueError, match="in_proj instead"):
        StrandsDeciderModel.from_pretrained_base(config)


def test_a_cuda_host_without_the_mamba_kernels_is_refused(base, monkeypatch):
    path, _, _ = base
    monkeypatch.setattr(M, "mamba_fallbacks", lambda: ["mamba2_chunk_scan"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="mamba-ssm"):
        StrandsDeciderModel._load_torso(_config(path), None, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="mamba-ssm"):
        M.require_mamba_kernels()


def test_nemotron_trains_reloads_and_serves(base, tmp_path):
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest

    path, rows, _ = base
    out = tmp_path / "out"
    T.train(
        T.TrainConfig(
            train_files=[str(tmp_path / "train.jsonl")],
            val_files=[str(tmp_path / "val.jsonl")],
            base_model=str(path),
            base_revision=None,
            pointer_dim=16,
            lora_targets=TARGETS,
            head_dropout=0.0,
            max_length=2048,
            micro_batch_size=4,
            grad_accum=2,
            max_steps=2,
            log_every=1,
            eval_every=0,
            gradient_checkpointing=True,
            output_dir=str(out),
        )
    )
    history = json.loads((out / "history.json").read_text())
    assert history and all(torch.isfinite(torch.tensor(h["loss"])) for h in history if "loss" in h)

    adapter = load_file(str(out / "lora" / "adapter_model.safetensors"))
    assert all("lora_" in k for k in adapter), "the adapter must not carry base weights"

    model = StrandsDeciderModel.load(str(out)).to(DEVICE).eval()
    trained = [n for n, p in model.torso.named_parameters() if "lora_B" in n and p.abs().sum() > 0]
    assert any("in_proj" in n for n in trained)  # the Mamba adapters trained

    state = " ".join(ex.state if isinstance(ex.state, str) else "x" for ex in rows[:3])
    qs = {
        f"q{i}": (
            NoulQuestion(instructions=rows[i].instructions + " ok?")
            if i % 2
            else ChoiceQuestion(
                instructions=rows[i].instructions, criteria={"a": None, "b": "bee", "c": None}
            )
        )
        for i in range(5)
    }
    req = SystemOneRequest(state=state, questions=qs)
    answers = [
        SystemOneEngine(model, EngineConfig(device=DEVICE, max_batch=4, use_prefix_cache=p))
        .evaluate(req)
        .answers
        for p in (True, False)
    ]
    for k in qs:
        x, y = answers[0][k].model_dump(), answers[1][k].model_dump()
        for f in x:
            if isinstance(x[f], dict):
                assert all(abs(x[f][o] - y[f][o]) <= 2e-3 for o in x[f]), (k, f)
            elif isinstance(x[f], float):
                assert abs(x[f] - y[f]) <= 2e-3, (k, f)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_the_mamba_kernels_run_on_the_decoders_device_not_the_current_one(base):
    """mamba-ssm and causal-conv1d launch on the current CUDA device; a decoder on another
    device computed wrong numbers or NaN until guard_mamba_devices (measured on the 30B)."""
    path, _, _ = base
    ids = torch.randint(4, 300, (2, 48))
    mask = torch.ones_like(ids)
    out = []
    for dev in ("cuda:0", "cuda:1"):
        torso = StrandsDeciderModel._load_torso(_config(path), dev, None).eval()
        assert all(
            getattr(m.forward, "on_input_device", False)
            for m in torso.modules()
            if hasattr(m, "time_step_limit")
        )
        with torch.no_grad():
            h = torso(input_ids=ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state
        out.append(h.float().cpu())
    assert torch.cuda.current_device() == 0
    assert torch.isfinite(out[1]).all()
    assert torch.allclose(out[0], out[1], atol=1e-5), (out[0] - out[1]).abs().max()
