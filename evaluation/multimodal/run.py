"""Image, video and audio evaluation of a strands-decider checkpoint on a Nemotron Omni base.

Every item goes through the serving path, `OmniEngine.evaluate` on a `SystemOneRequest`
with the media in `images` / `videos` / `audio` and the question as served (noul with its
default criteria, choice with the item's options), once with its media and once without
(the blind control: how much of the answer the media decide).

    python evaluation/multimodal/run.py CHECKPOINT --out DIR --data DIR [--sets ...]

Sets, each at a pinned revision or sha256 (all CC BY 4.0 or Apache-2.0; nothing is
redistributed):
- naturalbench: BaiqiL/NaturalBench, the first 300 groups (choice and yes/no on image pairs)
- pope: lmms-lab/POPE adversarial, the first 600 by question id (yes/no object presence)
- minds14: PolyAI/minds14 en-US, all 563 clips: which of 14 banking intents the caller wants
- speech_commands: the Speech Commands v0.02 test set, the 10 core words: "Does the speaker
  say 'w'?", each clip asked once about its own word and once about another
- perception: Perception Test MC validation (lmms-eval/PerceptionTest_Val), 3-way choice on
  short videos; the clips are read from the dataset's zips by range, not downloaded whole

Reuses evaluation/vision: its NaturalBench and POPE loaders, `run_system` (with resume) and
`score` (accuracy, Brier, ECE, NaturalBench's paired scores, media dependence).
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import random
import sys
import tarfile
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
VISION = os.path.join(HERE, "..", "vision")


def _vision_module(name: str) -> Any:
    """evaluation/vision/<name>.py, loaded by path: this file is also called run.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"vision_{name}", os.path.join(VISION, f"{name}.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, VISION)  # evaluation/vision/run.py imports its `metrics` module by name
_vrun = _vision_module("run")
summarise = _vrun.summarise
_versions, naturalbench, pope, run_system, score = (
    _vrun._versions,
    _vrun.naturalbench,
    _vrun.pope,
    _vrun.run_system,
    _vrun.score,
)

from strands_decider.media import MediaLimits  # noqa: E402
from strands_decider.omni import OmniEngineConfig, load_omni_engine  # noqa: E402
from strands_decider.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest  # noqa: E402

MINDS_REPO, MINDS_FILE, MINDS_REV = (
    "PolyAI/minds14",
    "en-US/train-00000-of-00001.parquet",
    "40ce77cb32a384e4d50a568e1ec39ac804019d33",
)
SC_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
SC_SHA256 = "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df"
SC_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
PT_REPO, PT_REV = "lmms-eval/PerceptionTest_Val", "c5e520d8c4167fb1f135c36e9d6e67312b4f8e6b"
PT_QUESTIONS = "mc_question_val/validation-00000-of-00001.parquet"
PT_ZIPS = ("videos_chunked_01.zip", "videos_chunked_02.zip")
# MINDS-14's intents with a short gloss each, as a caller's choice options.
MINDS_INTENTS = {
    "abroad": "using a card or the account abroad",
    "address": "changing the address on the account",
    "app_error": "a problem with the banking app",
    "atm_limit": "the cash machine withdrawal limit",
    "balance": "the account balance",
    "business_loan": "a loan for a business",
    "card_issues": "a problem with a card",
    "cash_deposit": "depositing cash",
    "direct_debit": "setting up or cancelling a direct debit",
    "freeze": "freezing or blocking the account or card",
    "high_value_payment": "making a large payment",
    "joint_account": "opening a joint account",
    "latest_transactions": "the latest transactions",
    "pay_bill": "paying a bill",
}


def _hub_file(repo: str, filename: str, revision: str, data: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo,
        filename,
        repo_type="dataset",
        revision=revision,
        local_dir=os.path.join(data, repo.replace("/", "__")),
    )


def minds14(data: str) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(_hub_file(MINDS_REPO, MINDS_FILE, MINDS_REV, data)).to_pylist()
    names = list(MINDS_INTENTS)
    for r in rows:
        intent = r["path"].split("~", 1)[1].split("/", 1)[0].lower()
        yield {
            "id": f"minds14-{os.path.basename(r['path'])}",
            "bench": "minds14",
            "kind": "choice",
            "question": "What does the caller want?",
            "options": [(n, MINDS_INTENTS[n]) for n in names],
            "gold": names.index(intent),
            "audio": r["audio"]["bytes"],
        }


def speech_commands(data: str, n_clips: int) -> Iterator[dict[str, Any]]:
    import hashlib
    import urllib.request

    path = os.path.join(data, os.path.basename(SC_URL))
    if not os.path.exists(path):
        urllib.request.urlretrieve(SC_URL, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != SC_SHA256:
        raise SystemExit(f"{path}: sha256 {digest}, expected {SC_SHA256}")
    rng = random.Random(0)
    with tarfile.open(path, "r:gz") as tar:
        members = sorted(
            (
                m
                for m in tar.getmembers()
                if m.isfile() and m.name.endswith(".wav") and m.name.split("/")[-2] in SC_WORDS
            ),
            key=lambda m: m.name,
        )
        rng.shuffle(members)
        per_word = collections.Counter()
        for m in members:
            word = m.name.split("/")[-2]
            if per_word[word] >= n_clips // len(SC_WORDS):
                continue
            per_word[word] += 1
            audio = tar.extractfile(m).read()  # type: ignore[union-attr]
            other = rng.choice([w for w in SC_WORDS if w != word])
            for asked, gold in ((word, 1), (other, 0)):
                yield {
                    "id": f"sc-{word}-{os.path.basename(m.name)}-{asked}",
                    "bench": "speech_commands",
                    "kind": "noul",
                    "question": f"Does the speaker say the word '{asked}'?",
                    "options": [("no", ""), ("yes", "")],
                    "gold": gold,
                    "audio": audio,
                    "word": word,
                }


def perception(data: str, n_questions: int) -> Iterator[dict[str, Any]]:
    import ast
    import zipfile

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    rows = pq.read_table(_hub_file(PT_REPO, PT_QUESTIONS, PT_REV, data)).to_pylist()
    by_video: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_video[r["video_name"]].append(r)
    videos = sorted(by_video)
    rng = random.Random(0)
    rng.shuffle(videos)
    picked: list[dict[str, Any]] = []
    for v in videos:  # two questions per video, so ~n/2 clips
        qs = by_video[v][:]
        rng.shuffle(qs)
        picked += qs[:2]
        if len(picked) >= n_questions:
            break
    picked = picked[:n_questions]
    vdir = os.path.join(data, "perception_videos")
    os.makedirs(vdir, exist_ok=True)
    need = {
        r["video_name"]
        for r in picked
        if not os.path.exists(os.path.join(vdir, r["video_name"] + ".mp4"))
    }
    if need:
        fs = HfFileSystem()
        for z in PT_ZIPS:
            with fs.open(f"datasets/{PT_REPO}@{PT_REV}/{z}", "rb", block_size=2**22) as fh:
                zf = zipfile.ZipFile(fh)
                for name in zf.namelist():
                    stem = os.path.basename(name).removesuffix(".mp4")
                    if stem in need:
                        with open(os.path.join(vdir, stem + ".mp4"), "wb") as out:
                            out.write(zf.read(name))
                        need.discard(stem)
        if need:
            raise SystemExit(f"Perception Test videos not found in the zips: {sorted(need)[:5]}")
    for r in picked:
        opts = r["options"]
        opts = ast.literal_eval(opts) if isinstance(opts, str) else list(opts)
        with open(os.path.join(vdir, r["video_name"] + ".mp4"), "rb") as fh:
            video = fh.read()
        yield {
            "id": f"pt-{r['video_name']}-{r['question_id']}",
            "bench": "perception",
            "kind": "choice",
            "question": r["question"],
            "options": [(o, "") for o in opts],
            "gold": int(r["answer_id"]),
            "video": video,
            "area": r["area"],
        }


class Omni:
    """The checkpoint served by OmniEngine; one request per item, the question as served."""

    def __init__(self, checkpoint: str, device: str, limits: MediaLimits, name: str) -> None:
        self.engine = load_omni_engine(checkpoint, OmniEngineConfig(device=device, limits=limits))
        self.name = name

    def probs(self, it: dict[str, Any], blind: bool) -> list[float]:
        if it["kind"] == "noul":
            q: Any = NoulQuestion(instructions=it["question"])
        else:
            q = ChoiceQuestion(
                instructions=it["question"], criteria={n: d or None for n, d in it["options"]}
            )
        media: dict[str, list[str]] = {}
        if not blind:
            for key, field in (("image", "images"), ("video", "videos"), ("audio", "audio")):
                if key in it:
                    media[field] = [base64.b64encode(it[key]).decode()]
        a = self.engine.evaluate(SystemOneRequest(state="", questions={"q": q}, **media)).answers[
            "q"
        ]
        if it["kind"] == "noul":
            p = float(a.noul)  # type: ignore[union-attr]
            return [1.0 - p, p]  # (no, yes): the items' option order
        return [float(a.probabilities[n]) for n, _ in it["options"]]  # type: ignore[union-attr]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--data", required=True, help="where downloads are kept")
    p.add_argument(
        "--sets",
        nargs="+",
        default=["naturalbench", "pope", "minds14", "speech_commands", "perception"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default="swiftpick")
    p.add_argument("--nb-groups", type=int, default=300)
    p.add_argument("--pope", type=int, default=600)
    p.add_argument("--sc-clips", type=int, default=300)
    p.add_argument("--pt-questions", type=int, default=300)
    p.add_argument("--image-token-budget", type=int, default=MediaLimits.image_token_budget)
    p.add_argument("--max-video-frames", type=int, default=MediaLimits.max_video_frames)
    p.add_argument(
        "--max-video-bytes",
        type=int,
        default=192 * 2**20,
        help="raised from the server's 64 MB: one Perception Test clip is 130 MB",
    )
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items: list[dict[str, Any]] = []
    for s in a.sets:
        if s == "naturalbench":
            items += list(naturalbench(a.nb_groups, None))
        elif s == "pope":
            items += list(pope(a.pope, None))
        elif s == "minds14":
            items += list(minds14(a.data))
        elif s == "speech_commands":
            items += list(speech_commands(a.data, a.sc_clips))
        elif s == "perception":
            items += list(perception(a.data, a.pt_questions))
        else:
            raise SystemExit(f"unknown set {s!r}")
    print(
        f"[multimodal-eval] {len(items)} items: {dict(collections.Counter(i['bench'] for i in items))}"
    )
    limits = replace(
        MediaLimits(),
        image_token_budget=a.image_token_budget,
        max_video_frames=a.max_video_frames,
        max_video_bytes=a.max_video_bytes,
    )
    system = Omni(a.checkpoint, a.device, limits, a.name)
    res = run_system(system, items, a.out)
    summary = {
        "checkpoint": a.checkpoint,
        "limits": limits.__dict__,
        "versions": _versions(),
        "datasets": {
            "naturalbench": "BaiqiL/NaturalBench@ba41a7d5",
            "pope": "lmms-lab/POPE@4db12766",
            "minds14": f"{MINDS_REPO}@{MINDS_REV[:8]}",
            "speech_commands": f"sha256 {SC_SHA256[:16]}",
            "perception": f"{PT_REPO}@{PT_REV[:8]}",
        },
        "scores": score(res),
    }
    area = {i["id"]: i["area"] for i in items if "area" in i}  # run_system keeps fixed fields only
    for tag, rs in res.items():
        by_area: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in rs:
            if r["id"] in area:
                by_area[area[r["id"]]].append(r)
        if by_area:
            summary["scores"][tag]["perception"]["by_area"] = {
                k: summarise(v)["all"] for k, v in sorted(by_area.items())
            }
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary["scores"], indent=1)[:3000])


if __name__ == "__main__":
    main()
