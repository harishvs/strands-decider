# Images, video and audio (swiftpick-30B-A3B)

swiftpick-30B-A3B answers questions about images, video and audio when it is served with
`--vision` (alias `--multimodal`). No weights change: the published adapter and pointer head
are used as is, and the media are read by the encoders that already ship inside its base,
Nemotron-3-Nano-Omni-30B-A3B: C-RADIOv4-H for images and video frames, Parakeet for audio.
The adapter was trained on text only, so this is zero-shot: see
[How well it does](#how-well-it-does) before relying on a modality. Text-only requests to a
multimodal server get the same answers as from a text server.

The Qwen3.5 image path is described in [vision.md](vision.md); the Gemma 4 models read text only.

## Using it

```bash
pip install "strands-decider[multimodal] @ git+https://github.com/harishvs/strands-decider@swiftpick-30B-A3B-v1-mm"
# Pillow and PyAV (FFmpeg); transformers 5.22; mamba-ssm and causal-conv1d kernels on CUDA
strands-decider serve harishvs/swiftpick-30B-A3B --multimodal --port 8000
```

A request adds any of `images`, `videos` and `audio`: lists of base64 files (a
`data:...;base64,` prefix is accepted). The state may be empty.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "A support call and the screen the caller is looking at.",
  "images": ["'"$(base64 < screen.png | tr -d '\n')"'"],
  "audio":  ["'"$(base64 < call.wav | tr -d '\n')"'"],
  "questions": {
    "angry":  {"type": "noul",   "instructions": "Is the caller upset?"},
    "intent": {"type": "choice", "instructions": "What does the caller want?",
               "criteria": {"refund": "money back", "fix": "the app to work", "other": null}}
  }
}'
```

From the command line: `strands-decider ask CHECKPOINT --image F --video F --audio F`
(each repeatable). `/health` reports `"modalities": ["text", "image", "video", "audio"]` and
the server's `media_limits`. A server started without `--multimodal` answers a request that
carries media with HTTP 422 rather than ignore them.

Formats and limits (each refusal is HTTP 422, before the model runs):

| | accepted | limits |
| --- | --- | --- |
| images | PNG, JPEG, WebP, GIF (first frame) | 4 per request, sharing a 2,048-token budget, at least 256 tokens each (1,024 to 13,312 patches of 16 px; one token per 2x2 patches); 4096 x 4096 px at most |
| video | MP4/MOV, MKV/WebM | 1 per request; at most one frame per second and 16 frames, spread evenly; 180 s, 64 MB; each frame resized to about 1,024 patches (~256 tokens per 2 frames); the audio track is not read |
| audio | WAV, FLAC, MP3, Ogg, M4A | 4 clips, 120 s together, at least 0.1 s each, 20 MB each; resampled to 16 kHz mono; about 12.5 tokens a second |

- Images and a video cannot share a request: the model's video and image inputs use the
  same placeholder token and the native model does not mix them. Either can go with audio.
- Everything counts against the same 4,096-token window as the state; an over-long prompt
  is refused. A 16-frame video uses about 2,100 tokens with its frame labels, two minutes of audio 1,500.
- The encoders and projectors add 1.44B parameters (2.9 GB in bf16): vision 0.65B, audio
  0.61B, projectors 0.18B. A multimodal server on one GPU used 66.9 GB after warm-up.

## How it works

`omni.load_omni` builds transformers' native `NemotronH_Omni_Reasoning_V3` (transformers
5.22, no NVIDIA remote code) from the base's own checkpoint and config, with three
documented fixes to the config: the audio sub-config's `model_type` key is dropped (the
native config rejects it), the absent `num_nextn_predict_layers` becomes 0, and EVS video
token pruning is off (the native code reads the wrong dimension). Loading refuses any
missing or unexpected weight beyond two known unused buffers.

The decoder inside it is the same `NemotronHModel` the text model loads, so the
checkpoint's LoRA adapter loads onto it unchanged. The language-model head is dropped; the
encoders and projectors are frozen. Two kernel fixes apply to both paths: Mamba's
`time_step_limit` is the config's, not a hard-coded one, and each Mamba layer runs on its
input's CUDA device, not the current one (a decoder on `cuda:1` otherwise produced NaN).

Preprocessing is a port of the native processors to PIL, torch and PyAV
(`strands_decider/media.py`), because they need torchvision and librosa. It was checked
once against them on real files: images to 4.8e-7, video frames exactly, audio log-mel to
4.5e-6 (`evaluation/multimodal/parity_media.py`).

Media go inside `<state>`, before the state text, in the model's own placeholder form:

```
<state>
<img><image>...<image></img>                                  (each image)
Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.00 seconds: <img><image>...</img>
Frame 3 sampled at 2.00 seconds and ...                       (a video, one chunk per 2 frames)
<so_start><so_embedding>...<so_embedding><so_end>             (each audio clip)
A support call and the screen the caller is looking at.
</state>
<question type="noul"> ... <answer>
```

The engine embeds the prompt, writes the encoders' outputs over the placeholder tokens, and
encodes that prefix once; each question is then a text suffix on a fork of its cache, as
for text. Nemotron's attention layers have no rotary positions, so nothing changes about
positions. A state or question that contains the placeholder tokens itself is refused.

Checks (`tests/test_omni.py`, `tests/test_media.py`, offline, on a tiny random native Omni
saved in the base's legacy checkpoint layout): the weights load exactly; text answers equal
the text engine's; for an image, two images, a video, audio, and image or video with audio,
the shared-prefix answers equal a plain full forward, and the spliced embeddings equal the
native model's; the refusals above; the grid, frame, label and token-count rules against
the native processors' own functions. On the real model: 200 text requests give identical
answers through the text and multimodal engines, and the base model with its head kept
captions images, transcribes speech and describes clips correctly through this
preprocessing.

## How well it does

Measured zero-shot (no media training) with the published checkpoint, through the serving
path (`OmniEngine.evaluate`, the default limits), on one H100 in bf16, once with the media and
once with them removed. Intervals are bootstrap 95% (NaturalBench by group, Perception Test
by video). Lower is better for Brier and ECE.

| modality | set | items | accuracy | Brier | ECE | media removed | chance | reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image | NaturalBench, first 300 groups | 1,200 | **0.820** (0.801-0.840) | 0.240 | 0.038 | 0.500 | 0.5 | Qwen3.5-2B v1 `--vision` 0.776 |
| image | POPE adversarial, first 600 | 600 | **0.870** (0.843-0.897) | 0.193 | 0.038 | 0.500 | 0.5 | Qwen3.5-2B v1 `--vision` 0.872 |
| audio | MINDS-14 en-US, 14 intents | 563 | **0.986** (0.975-0.995) | 0.026 | 0.026 | 0.082 | 0.071 | |
| audio | Speech Commands v0.02 test, keyword yes/no | 600 | **0.963** (0.948-0.978) | 0.090 | 0.135 | 0.500 | 0.5 | |
| video | Perception Test MC val, 16 frames | 300 (163 videos) | **0.720** (0.670-0.770) | 0.364 | 0.054 | 0.397 | 0.333 | |

- **Images:** on NaturalBench above the Qwen3.5-2B v1 release on the same 1,200 questions
  (G-Acc 0.417, Q-Acc 0.650, I-Acc 0.670; v1 G-Acc 0.310), and level with it on POPE.
- **Audio:** intent from a recorded banking call is near perfect (the 14 options carry a
  short gloss each). A spoken keyword is heard (0.943 when the clip
  says the word asked about, 0.983 when it does not), but those yes/no answers are
  underconfident: 24% of them put P(yes) between 0.20 and 0.80 (ECE 0.135).
- **Video:** well above chance; by Perception Test area: semantics 0.841 (113 questions), physics 0.779 (104), memory 0.80 (10), but
  abstraction 0.438 (73; 0.274 with the video removed).
- **Without the media it still answers confidently.** With the media removed, accuracy falls
  to chance, but the mean confidence stays at 0.74 (NaturalBench), 0.82 (POPE) and 0.82
  (Speech Commands). Its confidence does not tell you that a medium is missing or unreadable:
  check that it is there before you ask. On Perception Test the blind confidence drops to
  0.51.

Settings, on the same items: an image budget of 1,024 tokens instead of 2,048 gave 0.808 and
0.803 on the first 100 NaturalBench groups (no difference; 2,048 kept), and 8 video frames
instead of 16 gave 0.690 against 0.720 (16 minus 8: +0.030, 95% CI -0.003 to +0.063; 16 kept).

Sets: [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) (Apache-2.0) and
[POPE](https://huggingface.co/datasets/lmms-lab/POPE) at the revisions of
`evaluation/vision/run.py`; [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14)
(CC BY 4.0, PolyAI) at 40ce77cb; [Speech Commands v0.02](https://arxiv.org/abs/1804.03209)
(CC BY 4.0, Google) test set by sha256, the ten core words, each clip asked about its own
word and one other; [Perception Test](https://github.com/google-deepmind/perception_test)
(CC BY 4.0, Google DeepMind) multiple-choice validation via `lmms-eval/PerceptionTest_Val` at
c5e520d8, two questions per randomly drawn video. Nothing is redistributed. The command is
`evaluation/multimodal/run.py`; its eval-only `--max-video-bytes` (192 MB) admits the one
130 MB Perception Test clip that the server's 64 MB limit refuses.

## Limitations

- **Zero-shot.** The adapter and head never saw a media token; the numbers above are what
  the base's encoders and the text training give together. Other kinds of media (screens,
  documents, charts, music, long video) are not measured.
- **Video reads at most 16 frames**, about one a second: a long clip is sampled sparsely, and
  the video's audio track is ignored.
- **Abstract video questions are weak** (0.44 above).
- **Images and a video cannot be in one request.**
