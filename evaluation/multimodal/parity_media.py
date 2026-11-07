"""strands_decider.media against transformers' own nemotron_h_omni processors.

The serving environment has no torchvision or librosa, so media.py reproduces the native
image and video processors and ParakeetFeatureExtractor on PIL, torch and PyAV. This script
runs both on the same inputs, in an environment that has torchvision and librosa as well:

    python evaluation/multimodal/parity_media.py SNAPSHOT_DIR

SNAPSHOT_DIR is a local copy of the Omni base (for preprocessor_config.json and the
checkpoint's mel filterbank). Measured 2026-11-07 (torch 2.10, torchvision 0.25,
transformers 5.22.1, librosa 1.0): images max |diff| 4.8e-7 with identical grids, video 0,
audio 4.5e-6 with identical masks, mel filters 3.7e-9 from librosa's.
"""

import json
import sys

import librosa
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import ParakeetFeatureExtractor
from transformers.models.nemotron_h_omni.image_processing_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3ImageProcessor as IP,
)
from transformers.models.nemotron_h_omni.video_processing_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3VideoProcessor as VP,
)

from strands_decider import media as M

S = sys.argv[1]
pc = json.load(open(f"{S}/preprocessor_config.json"))
mean, std = torch.tensor(pc["norm_mean"]), torch.tensor(pc["norm_std"])
rng = np.random.default_rng(0)


def img(h, w):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 / max(w - 1, 1)), (y * 255 / max(h - 1, 1)), ((x + y) % 64) * 4], -1)
    a = np.clip(a + rng.normal(0, 25, a.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGB")


worst = 0.0
ip = IP(norm_mean=pc["norm_mean"], norm_std=pc["norm_std"])
for (h, w), budget in [
    ((448, 448), 13312),
    ((1080, 1920), 13312),
    ((300, 2000), 13312),
    ((2400, 1600), 13312),
    ((97, 133), 13312),
    ((1080, 1920), 4096),
    ((720, 1280), 2048),
    ((3000, 3000), 8192),
]:
    im = img(h, w)
    nat = ip(images=[im], max_num_patches=budget, return_tensors="pt")
    ours = M.preprocess_image(im, budget, mean, std)
    g = tuple(nat["image_grid_hw"][0].tolist())
    d = (
        (nat["pixel_values"] - ours.pixel_values).abs().max().item()
        if g == ours.grid_hw
        else float("nan")
    )
    worst = max(worst, d)
    print(f"image {h}x{w} budget {budget}: grid native {g} ours {ours.grid_hw} max|d| {d:.2e}")
vp = VP(norm_mean=pc["norm_mean"], norm_std=pc["norm_std"])
for h, w in [(48, 64), (720, 1280), (1080, 1920), (480, 360)]:
    frames = (
        torch.from_numpy(np.stack([np.asarray(img(h, w)) for _ in range(4)]))
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    nat = vp(videos=[frames], return_tensors="pt", do_sample_frames=False)
    ours = M.preprocess_video(frames, [0, 1, 2, 3], 2.0, mean, std)
    d = (nat["pixel_values_videos"] - ours.frames).abs().max().item()
    worst = max(worst, d)
    print(
        f"video {h}x{w}: shape native {tuple(nat['pixel_values_videos'].shape)} ours {tuple(ours.frames.shape)} tokens native {nat['num_tokens'][0]} ours {ours.tokens_per_tubelet} max|d| {d:.2e}"
    )
fe = ParakeetFeatureExtractor(feature_size=128, sampling_rate=16000)
for n in [1600, 16000, 16000 * 7 + 123, 16000 * 30]:
    t = np.arange(n) / 16000
    wave = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.normal(size=n)).astype(np.float32)
    nat = fe(wave, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    ours = M.audio_features(wave)
    m = nat["attention_mask"][0].bool()
    same_mask = torch.equal(m, ours.mask)
    d = (nat["input_features"][0] - ours.features).abs().max().item()
    worst = max(worst, d if same_mask else float("inf"))
    print(
        f"audio {n / 16000:.2f}s: frames native {nat['input_features'].shape[1]} ours {ours.features.shape[0]} mask equal {same_mask} max|d| {d:.2e}"
    )
lib = torch.from_numpy(
    librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000, norm="slaney")
).float()
idx = json.load(open(f"{S}/model.safetensors.index.json"))["weight_map"]
k = "sound_encoder.encoder.feature_extractor.featurizer.fb"
with safe_open(f"{S}/{idx[k]}", "pt") as f:
    fb = f.get_tensor(k).float()[0]
print(
    f"mel filters vs librosa max|d| {(M.mel_filters() - lib).abs().max():.2e}; vs checkpoint fb max|d| {(M.mel_filters() - fb).abs().max():.2e}; librosa vs fb {(lib - fb).abs().max():.2e}"
)
print(f"WORST preprocessing diff: {worst:.2e}")
