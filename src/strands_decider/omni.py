"""Image, video and audio input for checkpoints on nvidia/Nemotron-3-Nano-Omni-* bases.

The checkpoint's own Hub code does not run on the transformers this package supports, but
transformers itself has the whole model: `NemotronH_Omni_Reasoning_V3` with a RADIO vision
tower (images, and video as two-frame tubelets), a Parakeet audio encoder and their
projectors, and a converter for this checkpoint's weight names. This module loads it with
no Hub code and splits it in two:

- the torso is the model's text decoder (`language_model.model`, a `NemotronHModel`), the
  same module the text path loads, so a strands-decider adapter trained on text loads onto
  it unchanged and text answers are the text engine's;
- the encoders and projectors stay beside it, frozen. Their outputs replace the media
  placeholder tokens of the prompt's embeddings before the decoder reads them.

The adapter and head were trained on text only: media are read zero-shot.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn

from . import media as M
from .infer import EngineConfig, SystemOneEngine, UnforkableCache
from .modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    build_head,
    checkpoint_dir,
    config_path,
    fix_mamba_time_step_limit,
    guard_mamba_devices,
    load_head_state,
    require_mamba_kernels,
)
from .modeling import (
    base_revision as checkpoint_base_revision,
)
from .prompting import RenderedQuestion, render_question, render_state, to_answer
from .schema import Answer, Content, SystemOneRequest, SystemOneResponse, Usage

OMNI_MODEL_TYPE = "NemotronH_Nano_Omni_Reasoning_V3"  # the checkpoints' config.json model_type
IMAGE_TOKEN_ID = 18  # `<image>`, also video's: one `<img><image>*n</img>` per two-frame tubelet
AUDIO_TOKEN_ID = 27  # `<so_embedding>`
# What the checkpoint and transformers' class disagree on, and nothing else is allowed:
# RADIO's summary indices are a buffer the checkpoint lacks (they select cls tokens for
# `summary`, which this module never reads), and the checkpoint's input conditioner holds the
# image mean and std, which preprocessing applies (media.py) instead of the tower.
ALLOWED_MISSING = frozenset({"vision_model.summary_idxs"})
ALLOWED_UNEXPECTED = frozenset(
    {"vision_model.input_conditioner.norm_mean", "vision_model.input_conditioner.norm_std"}
)


def omni_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint's config.json as `NemotronH_Omni_Reasoning_V3_Config` accepts it.

    - Hub-code pointers (`auto_map`, `architectures`) and model types are dropped: the
      config names the classes. transformers 5.22's config also fails on
      `sound_config.model_type` ("parakeet" is not a registered type) because it pops it
      from a copy and rebuilds from the original.
    - `llm_config.num_nextn_predict_layers`, absent from the checkpoint, is set to 0 (no
      multi-token-prediction layers, as in the checkpoint) should a config carry a null.
    - `video_pruning_rate` becomes 0: transformers' video-token pruning reads the hidden
      size where it means the token grid (a bug), so video keeps every token.
    """
    d = copy.deepcopy(raw)
    for k in ("auto_map", "architectures", "model_type", "_name_or_path", "transformers_version"):
        d.pop(k, None)
    for sub in ("llm_config", "vision_config", "sound_config"):
        for k in ("auto_map", "architectures", "model_type"):
            (d.get(sub) or {}).pop(k, None)
    if (d.get("llm_config") or {}).get("num_nextn_predict_layers") is None:
        d.setdefault("llm_config", {})["num_nextn_predict_layers"] = 0
    d["video_pruning_rate"] = 0.0
    return d


def omni_config(base_model: str, revision: str | None) -> Any:
    """The native config of an Omni checkpoint, from its config.json, with no Hub code."""
    import transformers
    from transformers.utils import cached_file

    path = cached_file(base_model, "config.json", revision=revision)
    if path is None:
        raise FileNotFoundError(f"{base_model}: no config.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if raw.get("model_type") != OMNI_MODEL_TYPE:
        raise ValueError(
            f"{base_model}: model_type {raw.get('model_type')!r}, not {OMNI_MODEL_TYPE}"
        )
    cfg = transformers.NemotronH_Omni_Reasoning_V3_Config(**omni_config_dict(raw))
    if cfg.image_token_id != IMAGE_TOKEN_ID or cfg.audio_token_id != AUDIO_TOKEN_ID:
        raise ValueError(
            f"{base_model}: media token ids {cfg.image_token_id}, {cfg.audio_token_id}"
        )
    if (
        cfg.audio_config is None
        or getattr(cfg.vision_config, "video_temporal_patch_size", None) != 2
    ):
        raise ValueError(f"{base_model}: expected an audio encoder and two-frame video tubelets")
    return cfg


def load_omni(
    config: StrandsDeciderConfig, attn_implementation: str | None = None, device: str | None = None
) -> tuple[nn.Module, nn.Module]:
    """(torso, encoders) of an Omni base: the text decoder, and the frozen media modules.

    `encoders` is the native model with its `language_model` replaced by an empty module, so
    the decoder's parameters are held once (by the torso) and the unused LM head is freed.
    """
    import transformers

    cfg = omni_config(config.base_model, config.base_revision)
    require_mamba_kernels()
    kwargs: dict[str, Any] = {
        "config": cfg,
        "revision": config.base_revision,
        "dtype": getattr(torch, config.torch_dtype),
        "attn_implementation": attn_implementation or "sdpa",
        "output_loading_info": True,
    }
    if device:
        kwargs["device_map"] = device
    loaded = transformers.NemotronH_Omni_Reasoning_V3.from_pretrained(config.base_model, **kwargs)
    full, info = cast(tuple[Any, dict[str, Any]], loaded)
    missing = set(info["missing_keys"]) - ALLOWED_MISSING
    unexpected = set(info["unexpected_keys"]) - ALLOWED_UNEXPECTED
    if missing or unexpected or info.get("mismatched_keys"):
        raise RuntimeError(
            f"{config.base_model}: weights do not match the model: missing {sorted(missing)[:5]}, "
            f"unexpected {sorted(unexpected)[:5]}, mismatched {list(info.get('mismatched_keys', []))[:5]}"
        )
    torso: nn.Module = full.language_model.model
    fix_mamba_time_step_limit(torso)
    guard_mamba_devices(torso)
    torso.config.use_cache = True  # type: ignore[union-attr]
    full.language_model = nn.Identity()  # the torso holds the decoder; the LM head goes
    full.requires_grad_(False)
    full.eval()
    return torso, full


class OmniDeciderModel(StrandsDeciderModel):
    """A strands-decider checkpoint on an Omni base, with its media encoders beside the torso.

    Serving only: `load` attaches the checkpoint's text-trained adapter and head.
    """

    encoders: nn.Module
    image_mean: torch.Tensor
    image_std: torch.Tensor

    @classmethod
    def load(  # type: ignore[override]
        cls,
        path: str,
        *,
        attn_implementation: str | None = None,
        device: str | None = None,
    ) -> OmniDeciderModel:
        from peft import PeftModel
        from transformers import AutoTokenizer

        path = checkpoint_dir(path)
        config = StrandsDeciderConfig.from_json(config_path(path))
        config.base_revision = checkpoint_base_revision(path, config)
        lora_dir = os.path.join(path, "lora")
        if config.use_lora and not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"{lora_dir}: missing, but the checkpoint config sets use_lora")
        head_state = load_head_state(path)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        torso, encoders = load_omni(config, attn_implementation, device)
        if config.use_lora:
            torso = PeftModel.from_pretrained(torso, lora_dir, is_trainable=False)
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = config
        obj.torso = torso
        obj.tokenizer = tok
        obj.encoders = encoders
        obj.image_mean, obj.image_std = image_normalization(config.base_model, config.base_revision)
        obj.head = build_head(config, cls.hidden_size(torso))
        obj.head.load_state_dict(head_state)
        obj.head.to(torch.float32)
        return obj


def is_omni_base(base_model: str, revision: str | None) -> bool:
    """True if `base_model`'s config.json is an Omni checkpoint's (read with no Hub code)."""
    from transformers.utils import cached_file

    path = cached_file(base_model, "config.json", revision=revision)
    if path is None:
        return False
    with open(path, encoding="utf-8") as fh:
        return bool(json.load(fh).get("model_type") == OMNI_MODEL_TYPE)


def image_normalization(base_model: str, revision: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    """The vision tower's input mean and std, from the checkpoint's preprocessor_config.json
    (its `norm_mean` / `norm_std`; the tower itself expects normalized pixels)."""
    from transformers.utils import cached_file

    path = cached_file(base_model, "preprocessor_config.json", revision=revision)
    if path is None:
        raise FileNotFoundError(f"{base_model}: no preprocessor_config.json")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return torch.tensor(cfg["norm_mean"]), torch.tensor(cfg["norm_std"])


# ---------------------------------------------------------------- engine


@dataclass
class OmniEngineConfig(EngineConfig):
    limits: M.MediaLimits = field(default_factory=M.MediaLimits)


@dataclass
class Media:
    """A request's media, preprocessed, in prompt order: images or a video, then audio."""

    images: list[M.ImageInput] = field(default_factory=list)
    videos: list[M.VideoInput] = field(default_factory=list)
    audio: list[M.AudioInput] = field(default_factory=list)

    @property
    def visual_tokens(self) -> int:  # images and video share the `<image>` token
        return sum(i.n_tokens for i in self.images) + sum(v.n_tokens for v in self.videos)

    @property
    def audio_tokens(self) -> int:
        return sum(a.n_tokens for a in self.audio)

    def placeholders(self) -> str:
        blocks = [M.image_placeholder(i) for i in self.images]
        blocks += [M.video_placeholder(v) for v in self.videos]
        blocks += [M.audio_placeholder(a) for a in self.audio]
        return "\n".join(blocks)


def render_media_state(state: Content, media: Media) -> str:
    """`render_state(state)` with the media's expanded placeholders before the text.

    Media come first so the expensive part of the prefix (the encoders and the media
    tokens through the decoder) is identical across the questions of a request.
    """
    text = render_state(state)
    block = media.placeholders()
    if not block:
        return text
    body = text[len("<state>\n") : -len("\n</state>\n")]
    inner = f"{block}\n{body}" if body else block
    return f"<state>\n{inner}\n</state>\n"


class OmniEngine(SystemOneEngine):
    """Answers requests with images, video and audio on an Omni base; text as the text engine."""

    model: OmniDeciderModel

    def __init__(self, model: OmniDeciderModel, config: OmniEngineConfig | None = None):
        super().__init__(model, config or OmniEngineConfig())
        self.ocfg = cast(OmniEngineConfig, self.cfg)

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if not (request.images or request.videos or request.audio):
            return super().evaluate(request)
        media = self.prepare(request)
        names = list(request.questions)
        rendered = [render_question(request.questions[n]) for n in names]
        answers: dict[str, Answer] = {}
        total = 0
        for start in range(0, len(names), self.cfg.max_batch):
            chunk = rendered[start : start + self.cfg.max_batch]
            probs, ntok = self._media_probs(request.state, media, chunk)
            total += ntok
            for i, name in enumerate(names[start : start + self.cfg.max_batch]):
                rq = chunk[i]
                answers[name] = to_answer(
                    rq,
                    probs[i, : rq.n_slots].tolist(),
                    ordinal_smoothing=self.model.config.ordinal_smoothing,
                )
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=total, output_tokens=len(names)),
        )

    def prepare(self, request: SystemOneRequest) -> Media:
        """Decode and preprocess a request's media, refusing what the limits do not allow."""
        from .vision import decode_image

        lim = self.ocfg.limits
        for what, items, cap in (
            ("images", request.images, lim.max_images),
            ("videos", request.videos, lim.max_videos),
            ("audio clips", request.audio, lim.max_audio_clips),
        ):
            if len(items) > cap:
                raise ValueError(f"{len(items)} {what}; this server takes at most {cap}")
        if request.images and request.videos:
            # Both use the `<image>` placeholder, and the model reads one kind per request:
            # transformers' own forward fills every placeholder with the images, then again
            # with the video, and refuses a request that holds both.
            raise ValueError("a request carries images or a video, not both")
        mean, std = self.model.image_mean, self.model.image_std
        budget = M.image_patch_budget(len(request.images), lim.image_token_budget)
        images = [
            M.preprocess_image(decode_image(b, lim.max_image_pixels), budget, mean, std)
            for b in request.images
        ]
        videos = [M.preprocess_video(*M.decode_video(b, lim), mean, std) for b in request.videos]
        waves = [M.decode_audio(b, lim) for b in request.audio]
        seconds = sum(len(w) for w in waves) / M.SAMPLE_RATE
        if seconds > lim.max_audio_seconds:
            raise ValueError(f"{seconds:.0f} s of audio; at most {lim.max_audio_seconds:.0f} s")
        return Media(images, videos, [M.audio_features(w) for w in waves])

    def _fit_media(
        self, state: Content, media: Media, question_texts: list[str]
    ) -> tuple[list[int], list[list[int]], list[list[tuple[int, int]]]]:
        """Tokenise state (media expanded) and questions, refusing a prompt over the window or
        a state whose text itself holds media tokens."""
        max_len = self.model.config.max_length
        with self._tok_lock:
            enc = self.tok(question_texts, add_special_tokens=False, return_offsets_mapping=True)
            s = self.tok(render_media_state(state, media), add_special_tokens=True)["input_ids"]
        q, offs = enc["input_ids"], enc["offset_mapping"]
        if (
            s.count(IMAGE_TOKEN_ID) != media.visual_tokens
            or s.count(AUDIO_TOKEN_ID) != media.audio_tokens
        ):
            raise ValueError(
                "the state's text holds media placeholder tokens (<image>, <so_embedding>)"
            )
        if any(IMAGE_TOKEN_ID in x or AUDIO_TOKEN_ID in x for x in q):
            raise ValueError("a question holds media placeholder tokens (<image>, <so_embedding>)")
        longest = max(len(x) for x in q)
        if len(s) + longest > max_len:
            mt = media.visual_tokens + media.audio_tokens
            raise ValueError(
                f"prompt of {len(s) + longest} tokens ({len(s) - mt} state + {mt} media + "
                f"{longest} question) exceeds the context window of {max_len} tokens"
            )
        return s, q, list(offs)

    @torch.inference_mode()
    def media_embeds(self, prefix_ids: torch.Tensor, media: Media) -> torch.Tensor:
        """The prefix's input embeddings with each placeholder replaced by its media feature."""
        torso: Any = self.model.torso
        enc: Any = self.model.encoders
        emb: torch.Tensor = torso.get_input_embeddings()(prefix_ids)
        visual = []
        if media.images:
            pv = torch.cat([i.pixel_values for i in media.images]).to(self.device)
            grid = torch.tensor([i.grid_hw for i in media.images], device=self.device)
            visual += list(
                enc.get_image_features(pixel_values=pv, image_grid_hw=grid).pooler_output
            )
        for v in media.videos:
            out = enc.get_video_features(pixel_values_videos=v.frames.to(self.device)).pooler_output
            visual.append(out.reshape(-1, out.shape[-1]))
        for feats, token, what in (
            (visual, IMAGE_TOKEN_ID, "image/video"),
            (self._audio_features(media), AUDIO_TOKEN_ID, "audio"),
        ):
            if not feats:
                continue
            x = torch.cat(feats).to(emb.dtype)
            mask = prefix_ids == token
            if int(mask.sum()) != x.shape[0]:
                raise RuntimeError(f"{what}: {int(mask.sum())} placeholders, {x.shape[0]} features")
            emb = emb.masked_scatter(mask.unsqueeze(-1), x)
        return emb

    def _audio_features(self, media: Media) -> list[torch.Tensor]:
        enc: Any = self.model.encoders
        out = []
        for a in media.audio:  # one clip at a time: each has its own length
            feats = enc.get_audio_features(
                input_features=a.features.unsqueeze(0).to(self.device),
                input_features_mask=a.mask.unsqueeze(0).to(self.device),
            ).pooler_output
            if feats.shape[0] != a.n_tokens:
                raise RuntimeError(f"audio: {feats.shape[0]} embeddings, {a.n_tokens} expected")
            out.append(feats)
        return out

    def _media_probs(
        self, state: Content, media: Media, rendered: list[RenderedQuestion]
    ) -> tuple[torch.Tensor, int]:
        s, q, offsets = self._fit_media(state, media, [rq.text for rq in rendered])
        prefix_ids = torch.tensor([s], device=self.device)
        embeds = self.media_embeds(prefix_ids, media)
        try:
            return self._slot_probs_shared_prefix(
                s,
                q,
                [rq.n_slots for rq in rendered],
                [rq.kind for rq in rendered],
                offsets,
                rendered=rendered,
                prefix_embeds=embeds,
            )
        except UnforkableCache as e:  # media have no batched path: the prefix is the point
            raise RuntimeError(f"hybrid cache could not be forked: {e}") from e

    def ask_media(
        self,
        state: Content,
        questions: dict[str, Any],
        images: list[str] | None = None,
        videos: list[str] | None = None,
        audio: list[str] | None = None,
    ) -> SystemOneResponse:
        return self.evaluate(
            SystemOneRequest(
                state=state,
                questions=questions,
                images=images or [],
                videos=videos or [],
                audio=audio or [],
            )
        )


def load_omni_engine(
    checkpoint: str, config: OmniEngineConfig | None = None, attn_implementation: str | None = None
) -> OmniEngine:
    cfg = config or OmniEngineConfig()
    model = OmniDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    return OmniEngine(model, cfg)


def load_media_engine(
    checkpoint: str, config: EngineConfig | None = None, attn_implementation: str | None = None
) -> SystemOneEngine:
    """The engine that reads a checkpoint's media: OmniEngine (images, video, audio) on a
    Nemotron Omni base, else the Qwen3.5 vision engine (images)."""
    path = checkpoint_dir(checkpoint)
    cfg = StrandsDeciderConfig.from_json(config_path(path))
    engine_cfg = config or EngineConfig()
    if is_omni_base(cfg.base_model, checkpoint_base_revision(path, cfg)):
        omni_cfg = (
            engine_cfg
            if isinstance(engine_cfg, OmniEngineConfig)
            else OmniEngineConfig(**asdict(engine_cfg))
        )
        return load_omni_engine(path, omni_cfg, attn_implementation)
    from .vision import load_vision_engine

    return load_vision_engine(path, engine_cfg, attn_implementation=attn_implementation)


def modalities(engine: SystemOneEngine) -> list[str]:
    """What an engine reads, for /health."""
    if isinstance(engine, OmniEngine):
        return ["text", "image", "video", "audio"]
    from .vision import VisionEngine

    return ["text", "image"] if isinstance(engine, VisionEngine) else ["text"]
