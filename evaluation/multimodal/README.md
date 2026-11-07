# Multimodal evaluation

The measurements behind [docs/multimodal.md](../../docs/multimodal.md#how-well-it-does), on
swiftpick-30B-A3B (a Nemotron-3-Nano-Omni base). Each item goes through `OmniEngine.evaluate`
with its media and again without them.

```bash
pip install -e ".[multimodal]"   # plus the Mamba kernels, see the model card
C=checkpoints/swiftpick-30B-A3B   # a local copy of the checkpoint

# images and audio (2,963 items, ~25 min with the blind pass on one H100)
python evaluation/multimodal/run.py $C --out runs/mm/main --data data/mm \
  --sets naturalbench pope minds14 speech_commands
# video (300 questions; fetches ~160 clips, 2 GB, from the dataset's zips by range)
python evaluation/multimodal/run.py $C --out runs/mm/perception --data data/mm --sets perception

# the settings sweeps
python evaluation/multimodal/run.py $C --out runs/mm/img1024 --data data/mm \
  --sets naturalbench --nb-groups 100 --image-token-budget 1024
python evaluation/multimodal/run.py $C --out runs/mm/vid8 --data data/mm \
  --sets perception --max-video-frames 8
```

Each run writes `<name>.jsonl` and `<name>-blind.jsonl` (per-item probabilities; a rerun
resumes) and `summary.json` (accuracy, Brier, ECE per set, NaturalBench's paired scores,
Perception Test areas, media dependence, the limits and package versions).

`parity_media.py` is the one-time check of `strands_decider.media` against the native
transformers processors; it needs torchvision and librosa, so run it in a separate venv.
