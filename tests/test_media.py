"""strands_decider.media: decoding, limits and the Nemotron Omni preprocessing it reproduces.

Media are made in memory with PyAV, so nothing is downloaded. The expected grids, frame
labels and token counts are the values transformers' `nemotron_h_omni` processors produce
(their code needs torchvision and librosa, which the serving environment does not have).
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from strands_decider import media as M  # noqa: E402

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
LIMITS = M.MediaLimits()


def _video(
    container: str = "mp4",
    codec: str = "mpeg4",
    seconds: float = 3.0,
    fps: int = 10,
    size: tuple[int, int] = (64, 48),
) -> str:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format=container) as out:
        s = out.add_stream(codec, rate=fps)
        s.width, s.height = size
        s.pix_fmt = "yuv420p"
        for i in range(int(seconds * fps)):
            img = np.full((size[1], size[0], 3), (i * 8) % 256, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return base64.b64encode(buf.getvalue()).decode()


def _audio(
    container: str = "wav",
    codec: str = "pcm_s16le",
    seconds: float = 1.0,
    rate: int = 44100,
    channels: int = 2,
) -> str:
    buf = io.BytesIO()
    layout = "stereo" if channels == 2 else "mono"
    with av.open(buf, mode="w", format=container) as out:
        s = out.add_stream(codec, rate=rate, layout=layout)
        t = np.arange(int(seconds * rate)) / rate
        wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pcm = np.tile(wave, (channels, 1))
        fmt = s.codec_context.format.name
        dtype = {"s16": np.int16, "s32": np.int32, "flt": np.float32, "dbl": np.float64}[
            fmt.rstrip("p")
        ]
        if dtype in (np.int16, np.int32):
            pcm = pcm * np.iinfo(dtype).max
        pcm = pcm.astype(dtype)
        pcm = pcm if fmt.endswith("p") else pcm.T.reshape(1, -1)  # planar or packed
        frame = av.AudioFrame.from_ndarray(pcm, format=fmt, layout=layout)
        frame.sample_rate = rate
        for pkt in s.encode(frame):
            out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- images


def _native(name: str):
    """A pure-math function from transformers' nemotron_h_omni processors, whose modules need
    torchvision to import: its source, run in a namespace that has `math` only."""
    import ast
    import math

    import transformers.models.nemotron_h_omni as pkg

    root = pkg.__path__[0]
    for f in ("image_processing_nemotron_h_omni.py", "video_processing_nemotron_h_omni.py"):
        src = open(f"{root}/{f}").read()
        tree = ast.parse(src)
        nodes = [*tree.body, *(n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body)]
        for node in nodes:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                ns: dict = {"math": math}
                exec(compile(ast.Module([node], []), f, "exec"), ns)
                return ns[name]
    raise LookupError(name)


@pytest.mark.parametrize(
    "h, w, hi",
    [
        (448, 448, 13312),
        (1080, 1920, 13312),
        (4096, 4096, 13312),
        (100, 3000, 13312),
        (1080, 1920, 4096),
        (37, 4000, 8192),
        (3000, 17, 2048),
        (512, 512, 1100),
    ],
)
def test_the_patch_grid_is_the_native_processors(h, w, hi):
    native = _native("get_patch_grid_size")(h, w, 16, 2, 1024, hi)
    assert M.patch_grid(h, w, 1024, hi) == native
    gh, gw = native
    assert gh % 2 == 0 and gw % 2 == 0 and gh * gw <= hi


def test_a_small_image_still_gets_the_floor_when_the_budget_is_the_floor():
    gh, gw = M.patch_grid(30, 40, 1024, 1024)  # the native rule would keep 2x4 patches here
    assert 1024 <= gh * gw <= 1024 + 64 and gh % 2 == 0 and gw % 2 == 0


def test_images_share_the_token_budget_within_the_models_range():
    assert M.image_patch_budget(1, 2048) == 8192
    assert M.image_patch_budget(4, 2048) == 2048
    assert M.image_patch_budget(16, 2048) == M.MIN_IMAGE_PATCHES  # never under 256 tokens
    assert M.image_patch_budget(1, 10**6) == M.MAX_IMAGE_PATCHES


def test_an_image_becomes_normalized_channel_major_patches():
    img = Image.new("RGB", (40, 30), (255, 0, 0))
    out = M.preprocess_image(img, 1024, MEAN, STD)
    gh, gw = out.grid_hw
    assert out.pixel_values.shape == (gh * gw, 3 * 16 * 16) and gh * gw >= 1024
    assert out.n_tokens == gh * gw // 4
    first = out.pixel_values[0].view(3, 16, 16)
    assert torch.allclose(first[0], (1 - MEAN[0]) / STD[0], atol=1e-3)  # red channel first
    assert torch.allclose(first[1], (0 - MEAN[1]) / STD[1], atol=1e-3)
    assert M.image_placeholder(out) == "<img>" + "<image>" * out.n_tokens + "</img>"


# ---------------------------------------------------------------- video


@pytest.mark.parametrize(
    "container, codec", [("mp4", "mpeg4"), ("mp4", "libx264"), ("webm", "libvpx-vp9")]
)
def test_a_video_is_sampled_at_most_once_a_second(container, codec):
    frames, idx, fps = M.decode_video(_video(container, codec, seconds=3.0, fps=10), LIMITS)
    assert fps == 10.0 and frames.dtype == torch.uint8 and frames.shape[1] == 3
    assert idx == [0, 14, 29] and frames.shape[0] == 3


def test_only_the_sampled_frames_are_decoded_and_they_are_the_videos_own(monkeypatch):
    data = _video(seconds=4.0, fps=10)
    frames, idx, _ = M.decode_video(data, LIMITS)
    with av.open(io.BytesIO(base64.b64decode(data))) as c:
        every = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    assert len(every) == 40 and idx == M.sample_frame_indices(40, 10.0, 16)
    assert torch.equal(
        frames, torch.from_numpy(np.stack([every[i] for i in idx])).permute(0, 3, 1, 2)
    )

    passes = []
    real = M._video_pass
    monkeypatch.setattr(M, "_video_pass", lambda raw, keep: passes.append(keep) or real(raw, keep))
    with pytest.raises(ValueError, match="at most 2 s"):
        M.decode_video(data, M.MediaLimits(max_video_seconds=2.0))
    assert passes == [None]  # refused from the packet count, nothing decoded


def test_a_long_video_keeps_at_most_max_frames_spread_evenly():
    idx = M.sample_frame_indices(n_frames=600, fps=10.0, max_frames=16)
    assert len(idx) == 16 and idx[0] == 0 and idx[-1] == 599
    assert M.sample_frame_indices(5, 30.0, 16) == [0]  # under a second: one frame


def test_video_frames_and_labels_are_the_native_processors():
    class _P:  # the native method reads these attributes
        video_target_num_patches, _downsample_factor, video_maintain_aspect_ratio = 1024, 2, True

    native = _native("_compute_target_patches")
    for h, w in ((48, 64), (1080, 1920), (1920, 1080), (720, 1280), (100, 3000), (64, 64)):
        pw, ph = native(_P(), h, w)
        assert M.video_tile(h, w) == (ph, pw), (h, w)
    assert M.video_tile(48, 64) == (28, 36)
    frames = torch.zeros(3, 3, 48, 64, dtype=torch.uint8)
    v = M.preprocess_video(frames, [0, 14, 29], 10.0, MEAN, STD)
    assert v.frames.shape == (3, 3, 28 * 16, 36 * 16)
    assert v.tokens_per_tubelet == 28 * 36 // 4 and v.n_tubelets == 2
    group = "<img>" + "<image>" * v.tokens_per_tubelet + "</img>"
    assert M.video_placeholder(v) == (
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.40 seconds: " + group + "\n"
        "Frame 3 sampled at 2.90 seconds: " + group
    )


def test_video_refusals():
    gif = io.BytesIO()
    frames = [Image.new("P", (32, 32), i) for i in range(3)]
    frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:], duration=200)
    with pytest.raises(ValueError, match="format 'gif' is not accepted"):
        M.decode_video(base64.b64encode(gif.getvalue()).decode(), LIMITS)
    with pytest.raises(ValueError, match="could not be read"):
        M.decode_video(base64.b64encode(b"not a video at all" * 10).decode(), LIMITS)
    with pytest.raises(ValueError, match="not valid base64"):
        M.decode_video("@@@", LIMITS)
    with pytest.raises(ValueError, match="larger than"):
        M.decode_video(_video(seconds=1.0), M.MediaLimits(max_video_bytes=100))
    with pytest.raises(ValueError, match="at most 2 s"):
        M.decode_video(_video(seconds=3.0), M.MediaLimits(max_video_seconds=2.0))


# ---------------------------------------------------------------- audio


@pytest.mark.parametrize(
    "container, codec",
    [("wav", "pcm_s16le"), ("flac", "flac"), ("mp3", "libmp3lame"), ("ogg", "libopus")],
)
def test_audio_is_resampled_to_16khz_mono(container, codec):
    wave = M.decode_audio(_audio(container, codec, seconds=1.0, rate=48000), LIMITS)
    assert wave.dtype == np.float32 and wave.ndim == 1
    assert abs(len(wave) - 16000) <= 1200  # codec padding and priming
    assert 0.1 < np.abs(wave).max() < 0.5


def test_the_audio_token_count_is_the_encoders_subsampled_length():
    from transformers import ParakeetEncoder, ParakeetEncoderConfig

    enc = ParakeetEncoder(
        ParakeetEncoderConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=8,
            subsampling_conv_channels=4,
            num_mel_bins=16,
        )
    )
    for n in (1600, 16000, 16001, 48000, 160 * 1000 + 37):
        frames = n // M.HOP  # the mask's real frames
        assert M.audio_tokens(n) == int(
            enc._get_subsampling_output_length(torch.tensor(frames + 1))
        )


def test_audio_features_are_normalized_log_mel_frames():
    t = np.arange(16000) / 16000
    out = M.audio_features((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    assert out.features.shape == (1 + 16000 // M.HOP, M.N_MELS)
    assert int(out.mask.sum()) == 16000 // M.HOP and not bool(out.mask[-1])
    real = out.features[out.mask]
    assert torch.allclose(real.mean(0), torch.zeros(M.N_MELS), atol=1e-4)
    assert float(real.std(0).mean()) == pytest.approx(1.0, abs=0.05)
    assert out.n_tokens == M.audio_tokens(16000)
    assert M.audio_placeholder(out) == "<so_start>" + "<so_embedding>" * out.n_tokens + "<so_end>"


def test_mel_filters_are_slaney_and_shaped_for_the_encoder():
    fb = M.mel_filters()
    assert fb.shape == (128, 257) and fb.dtype == torch.float32
    assert bool((fb >= 0).all()) and bool((fb.sum(1) > 0).all())


def test_audio_refusals():
    with pytest.raises(ValueError, match="shorter than"):
        M.decode_audio(_audio(seconds=0.05), LIMITS)
    with pytest.raises(ValueError, match="could not be read"):
        M.decode_audio(base64.b64encode(b"garbage" * 50).decode(), LIMITS)
    with pytest.raises(ValueError, match="larger than"):
        M.decode_audio(_audio(seconds=1.0), M.MediaLimits(max_audio_bytes=100))
    data = "data:audio/wav;base64," + _audio(seconds=0.5)
    assert len(M.decode_audio(data, LIMITS)) > 7000  # a data URI is accepted
