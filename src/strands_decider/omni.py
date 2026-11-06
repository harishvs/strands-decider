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
from typing import Any, cast

import torch
import torch.nn as nn

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
    - `llm_config.num_nextn_predict_layers` is null in the checkpoint; the class compares
      it with 0, so it becomes 0 (no multi-token-prediction layers, as in the checkpoint).
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
