"""Image, video and audio inputs as Nemotron-3-Nano-Omni reads them, with no torchvision or librosa.

Each function reproduces a transformers 5.22 preprocessor of the `nemotron_h_omni` model
(Apache-2.0, The HuggingFace Team) on PIL, torch and PyAV:

- images: `NemotronH_Omni_Reasoning_V3ImageProcessor`: a dynamic-resolution patch grid of
  16-pixel patches, bicubic antialiased resize in float32, CLIP mean/std, channel-major
  patches. One token per 2x2 patches after the projector's pixel shuffle.
- video: `NemotronH_Omni_Reasoning_V3VideoProcessor`: frames resized to one aspect-preserving
  tile near 1,024 patches; the tower reads them two at a time (a tubelet), one
  `<img>...</img>` chunk per tubelet with the processor's frame/timestamp labels.
- audio: `ParakeetFeatureExtractor`: 16 kHz mono, preemphasis 0.97, a 400-sample Hann
  window, 512-point STFT, hop 160, 128 slaney mel bands, log, per-clip normalization.

Media arrive base64-encoded (a `data:...;base64,` prefix is accepted). Every decoder checks
its size, format, duration and count limits and raises ValueError, which the server answers
with 422. The audio track of a video is not read.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PATCH = 16
MERGE = 2  # the projector's pixel shuffle: 2x2 patches per token
MIN_IMAGE_PATCHES = 1024  # the model's floor: at least 256 tokens per image
MAX_IMAGE_PATCHES = 13312
VIDEO_TARGET_PATCHES = 1024
TUBELET = 2  # frames per video embedding step
SAMPLE_RATE = 16000
HOP, N_FFT, WIN, N_MELS = 160, 512, 400, 128
PREEMPHASIS = 0.97
LOG_GUARD = 2.0**-24
NORM_EPS = 1e-5
# FFmpeg demuxer names (`container.format.name`) this module opens, and nothing else.
VIDEO_CONTAINERS = ("mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm")
AUDIO_CONTAINERS = ("wav", "flac", "mp3", "ogg", "mov,mp4,m4a,3gp,3g2,mj2")

IMG_START, IMG, IMG_END = "<img>", "<image>", "</img>"
AUDIO_START, AUDIO, AUDIO_END = "<so_start>", "<so_embedding>", "<so_end>"


@dataclass(frozen=True)
class MediaLimits:
    """What one request may carry; the engine's window bounds the total as well."""

    max_images: int = 4
    image_token_budget: int = 2048  # shared by a request's images; at least 256 each
    max_image_pixels: int = 4096 * 4096
    max_videos: int = 1
    max_video_frames: int = 16  # even: two frames per tubelet
    max_video_seconds: float = 180.0
    max_video_bytes: int = 64 * 2**20
    max_audio_clips: int = 4
    max_audio_seconds: float = 120.0  # all clips together
    min_audio_seconds: float = 0.1
    max_audio_bytes: int = 20 * 2**20


@dataclass
class ImageInput:
    pixel_values: torch.Tensor  # (patches, 3 * 16 * 16), normalized
    grid_hw: tuple[int, int]

    @property
    def n_tokens(self) -> int:
        return self.grid_hw[0] * self.grid_hw[1] // MERGE**2


@dataclass
class VideoInput:
    frames: torch.Tensor  # (frames, 3, H, W), normalized
    indices: list[int]  # sampled frame numbers in the source video
    fps: float
    tokens_per_tubelet: int

    @property
    def n_tubelets(self) -> int:
        return (len(self.indices) + TUBELET - 1) // TUBELET

    @property
    def n_tokens(self) -> int:
        return self.n_tubelets * self.tokens_per_tubelet


@dataclass
class AudioInput:
    features: torch.Tensor  # (frames, 128) log-mel, normalized
    mask: torch.Tensor  # (frames,) the real mel frames
    seconds: float
    n_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_tokens = audio_tokens(round(self.seconds * SAMPLE_RATE))


# ---------------------------------------------------------------- base64


def decode_base64(data: str, what: str, max_bytes: int) -> bytes:
    """The bytes of a base64 payload (a data URI's prefix is dropped), refused past max_bytes."""
    if data.startswith("data:"):
        head, _, data = data.partition(",")
        if ";base64" not in head:
            raise ValueError(f"{what} data URI must be base64-encoded")
    if len(data) * 3 // 4 > max_bytes:
        raise ValueError(f"{what} is larger than {max_bytes // 2**20} MB")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"{what} is not valid base64: {e}") from e


# ---------------------------------------------------------------- images


def patch_grid(
    height: int,
    width: int,
    min_patches: int = MIN_IMAGE_PATCHES,
    max_patches: int = MAX_IMAGE_PATCHES,
) -> tuple[int, int]:
    """The (rows, cols) patch grid an image is resized to: near its own, scaled into
    [min_patches, max_patches], both sides even (the native `get_patch_grid_size`).

    One difference: the native function scales a small image up only when max_patches
    exceeds min_patches, which its budget always does. A request's shared budget can equal
    the floor, and a small image would then get a handful of tokens; here it still scales
    up to min_patches."""
    gh = round(height / PATCH + 0.5)
    gw = round(width / PATCH + 0.5)
    factor = min(math.sqrt(max_patches / (gh * gw)), 1.0)
    gh, gw = math.floor(factor * gh), math.floor(factor * gw)
    if max_patches >= min_patches and gh * gw < min_patches:
        factor = math.sqrt(min_patches / (gh * gw))
        gh, gw = math.ceil(factor * gh), math.ceil(factor * gw)
    if rem := gh % MERGE:
        gh = gh + MERGE - rem if (gh + MERGE - rem) * gw <= max_patches else max(MERGE, gh - rem)
    if rem := gw % MERGE:
        gw = gw + MERGE - rem if gh * (gw + MERGE - rem) <= max_patches else max(MERGE, gw - rem)
    return gh, gw


def image_patch_budget(n_images: int, token_budget: int) -> int:
    """Patches each of a request's images may use: the shared token budget split evenly,
    clamped to the model's range."""
    per = (token_budget * MERGE**2) // max(n_images, 1)
    return max(MIN_IMAGE_PATCHES, min(MAX_IMAGE_PATCHES, per))


def _normalize(pixels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (pixels / 255.0 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def _resize(pixels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bicubic antialiased resize in float32 (torchvision's tensor resize is the same op)."""
    if pixels.shape[-2:] == (height, width):
        return pixels
    return F.interpolate(
        pixels, size=(height, width), mode="bicubic", align_corners=False, antialias=True
    )


def to_patches(pixels: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) to (patches, 3 * 16 * 16): row-major patches, each channel-major."""
    _, c, h, w = pixels.shape
    p = pixels.reshape(1, c, h // PATCH, PATCH, w // PATCH, PATCH).permute(0, 2, 4, 1, 3, 5)
    return p.reshape((h // PATCH) * (w // PATCH), c * PATCH * PATCH)


def preprocess_image(
    img: Any, max_patches: int, mean: torch.Tensor, std: torch.Tensor
) -> ImageInput:
    """A decoded RGB PIL image as the vision tower reads it."""
    pixels = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.uint8).copy())
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    gh, gw = patch_grid(pixels.shape[-2], pixels.shape[-1], MIN_IMAGE_PATCHES, max_patches)
    pixels = _normalize(_resize(pixels, gh * PATCH, gw * PATCH), mean, std)
    return ImageInput(to_patches(pixels), (gh, gw))


def image_placeholder(image: ImageInput) -> str:
    return IMG_START + IMG * image.n_tokens + IMG_END


# ---------------------------------------------------------------- video


def _av() -> Any:
    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'video and audio input need PyAV: pip install "strands-decider[multimodal]"'
        ) from e
    return av


def _open(raw: bytes, allowed: tuple[str, ...], what: str) -> Any:
    av = _av()
    try:
        container = av.open(io.BytesIO(raw), mode="r")
    except Exception as e:  # FFmpeg raises its own error types
        raise ValueError(f"{what} could not be read: {e}") from e
    if container.format.name not in allowed:
        name = container.format.name
        container.close()
        raise ValueError(f"{what} format {name!r} is not accepted")
    return container


def sample_frame_indices(n_frames: int, fps: float, max_frames: int) -> list[int]:
    """At most one frame per second and at most max_frames, spread evenly over the clip."""
    want = min(max_frames, max(1, round(n_frames / max(fps, 1e-6))), n_frames)
    return sorted({int(np.rint(x)) for x in np.linspace(0, n_frames - 1, want)})


def video_tile(height: int, width: int) -> tuple[int, int]:
    """(patch rows, patch cols) of every frame: an aspect-preserving tile near 1,024 patches,
    both sides even (the native `_compute_target_patches`)."""
    aspect = width / max(height, 1)
    ph = max(round(math.sqrt(VIDEO_TARGET_PATCHES / aspect)), 1)
    pw = max(round(math.sqrt(VIDEO_TARGET_PATCHES * aspect)), 1)
    rh, rw = ph % MERGE, pw % MERGE
    ph_up, pw_up = ph + (MERGE - rh if rh else 0), pw + (MERGE - rw if rw else 0)
    if ph_up * pw_up <= VIDEO_TARGET_PATCHES:
        return ph_up, pw_up
    return max(MERGE, ph - rh), max(MERGE, pw - rw)


def decode_video(data: str, limits: MediaLimits) -> tuple[torch.Tensor, list[int], float]:
    """(frames as uint8 (n, 3, H, W), their indices, source fps) of a base64 video."""
    raw = decode_base64(data, "video", limits.max_video_bytes)
    container = _open(raw, VIDEO_CONTAINERS, "video")
    try:
        if not container.streams.video:
            raise ValueError("video has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0) or 25.0
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"video could not be decoded: {e}") from e
    finally:
        container.close()
    if not frames:
        raise ValueError("video has no frames")
    if len(frames) / fps > limits.max_video_seconds:
        raise ValueError(
            f"video is {len(frames) / fps:.0f} s; at most {limits.max_video_seconds:.0f} s"
        )
    indices = sample_frame_indices(len(frames), fps, limits.max_video_frames)
    picked = np.stack([frames[i] for i in indices])  # (n, H, W, 3)
    return torch.from_numpy(picked).permute(0, 3, 1, 2).contiguous(), indices, fps


def preprocess_video(
    frames: torch.Tensor, indices: list[int], fps: float, mean: torch.Tensor, std: torch.Tensor
) -> VideoInput:
    ph, pw = video_tile(frames.shape[-2], frames.shape[-1])
    x = _normalize(_resize(frames.to(torch.float32), ph * PATCH, pw * PATCH), mean, std)
    return VideoInput(x, indices, fps, ph * pw // MERGE**2)


def video_placeholder(video: VideoInput) -> str:
    """One labelled `<img>...</img>` chunk per tubelet, as the native processor writes them."""
    group = IMG_START + IMG * video.tokens_per_tubelet + IMG_END
    frame_ms = int(1000.0 / video.fps)
    lines = []
    for g in range(video.n_tubelets):
        parts = []
        for off in range(TUBELET):
            i = g * TUBELET + off
            if i >= len(video.indices):
                break
            ts = int(video.indices[i]) * frame_ms / 1000.0
            parts.append(f"{'Frame' if off == 0 else 'frame'} {i + 1} sampled at {ts:.2f} seconds")
        lines.append(" and ".join(parts) + ": " + group)
    return "\n".join(lines)


# ---------------------------------------------------------------- audio


def decode_audio(data: str, limits: MediaLimits) -> np.ndarray:
    """A base64 audio file as 16 kHz mono float32 samples."""
    av = _av()
    raw = decode_base64(data, "audio", limits.max_audio_bytes)
    container = _open(raw, AUDIO_CONTAINERS, "audio")
    try:
        if not container.streams.audio:
            raise ValueError("audio has no audio stream")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"audio could not be decoded: {e}") from e
    finally:
        container.close()
    wave = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    if len(wave) < limits.min_audio_seconds * SAMPLE_RATE:
        raise ValueError(f"audio is shorter than {limits.min_audio_seconds} s")
    return wave


def audio_tokens(n_samples: int) -> int:
    """Embeddings the audio encoder emits for a clip: center-padded STFT frames, then three
    stride-2 convolutions (kernel 3, padding 1), as the native processor estimates."""
    length = 1 + n_samples // HOP
    for _ in range(3):
        length = (length + 2 - 3) // 2 + 1
    return max(1, length)


_MEL: dict[str, torch.Tensor] = {}


def mel_filters() -> torch.Tensor:
    """(128, 257) slaney mel filters for 16 kHz audio and a 512-point FFT."""
    if "mel" not in _MEL:
        from transformers.audio_utils import mel_filter_bank

        fb = mel_filter_bank(
            num_frequency_bins=N_FFT // 2 + 1,
            num_mel_filters=N_MELS,
            min_frequency=0.0,
            max_frequency=SAMPLE_RATE / 2,
            sampling_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        )
        _MEL["mel"] = torch.from_numpy(np.asarray(fb, dtype=np.float64).T).to(torch.float32)
    return _MEL["mel"]


def audio_features(wave: np.ndarray) -> AudioInput:
    """Log-mel features of one 16 kHz clip, normalized over its real frames."""
    x = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))
    x = torch.cat([x[:1], x[1:] - PREEMPHASIS * x[:-1]])
    stft = torch.stft(
        x.unsqueeze(0),
        N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=torch.hann_window(WIN, periodic=False),
        return_complex=True,
        pad_mode="constant",
    )
    power = torch.view_as_real(stft).pow(2).sum(-1).sqrt().pow(2)
    mel = torch.log(mel_filters() @ power + LOG_GUARD)[0].T  # (frames, 128)
    n = len(wave) // HOP  # the frames the native mask keeps
    mask = torch.arange(mel.shape[0]) < n
    real = mel[:n]
    mean = real.mean(0, keepdim=True)
    std = real.var(0, unbiased=True, keepdim=True).sqrt() if n > 1 else torch.zeros_like(mean)
    feats = ((mel - mean) / (std + NORM_EPS)) * mask.unsqueeze(-1)
    return AudioInput(feats, mask, len(wave) / SAMPLE_RATE)


def audio_placeholder(audio: AudioInput) -> str:
    return AUDIO_START + AUDIO * audio.n_tokens + AUDIO_END
