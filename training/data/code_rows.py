"""Code-task rows built from public sources: five yes/no judgements and one choice.

    python -m training.data.code_rows --cubert DIR --codereviewer DIR \\
        --out data/code_noul_v1.jsonl --eval-out data/code_noul_v1_eval.jsonl

The Gemma 4 v1 recipe trains on 12,263 code-task rows (`data/code_v27.jsonl`) that are built
outside this repository and not published. This builder makes a public stand-in for the part
of them that `data/sources.md` describes as yes/no judgements, to test whether such rows help a
torso's yes/no transfer. It is not v27: it leaves out CommitPackFT, CodeContests, Juliet and
the generated programs.

Sources, pinned by the caller's download (`hf download ... --revision`):
  - `claudios/cubert_ETHPy150Open` (Apache-2.0), revision 0e383204f0d0c59b1ba4e89fff82c20e52e8d004:
    function_docstring, variable_misuse, swapped_operands, wrong_binary_operator and exception,
    `train` and `test` parquet files.
  - `fasterinnerlooper/codereviewer` (a mirror of Zenodo 6900648, CC BY 4.0), revision
    555f3cda18bb62f90e322bb6b67d51b4a6814af5: `quality/cls-valid.jsonl` (training rows) and
    `quality/cls-test.jsonl` (held-out rows).

Training rows come only from train/validation files and held-out rows only from test files;
a held-out row whose state also occurs in training (a short function in both) is dropped.
CuBERT stores a function's original next to its mutated copies, so rows are drawn one per
function (its `info` path without the variant suffix): no function is seen both ways. Each
yes/no task is balanced 50/50. Every choice is seeded, so a rebuild gives the same file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections.abc import Iterable, Iterator
from typing import Any

from training.data.format import Example, write_jsonl

# task -> (CuBERT config, the label that means "yes", question, variants, false/true texts)
CUBERT_NOUL: dict[str, tuple[str, str, str, list[str], str, str]] = {
    "code_docstring_match": (
        "function_docstring_datasets",
        "Correct",
        "Does the docstring describe this function?",
        [
            "Is this docstring the one written for this function?",
            "Does the documentation match what the function does?",
            "Is the given docstring an accurate description of this function?",
            "Does this function belong with this docstring?",
        ],
        "the docstring belongs to a different function",
        "the docstring describes this function",
    ),
    "code_variable_misuse": (
        "variable_misuse_datasets",
        "Variable misuse",
        "Does this function use the wrong variable somewhere?",
        [
            "Is a variable used where another one was meant?",
            "Does this code contain a variable-misuse bug?",
            "Has one variable been mistakenly used in place of another here?",
            "Is there a place where this function reads or writes the wrong variable?",
        ],
        "every variable is the intended one",
        "a variable is used where another was intended",
    ),
    "code_swapped_operands": (
        "swapped_operands_datasets",
        "Swapped operands",
        "Are the two operands of a binary operation swapped somewhere in this function?",
        [
            "Does this code contain a swapped-operands bug?",
            "Has the order of two operands been reversed by mistake?",
            "Is there an operation whose left and right sides are in the wrong order?",
            "Are two operands in this function the wrong way round?",
        ],
        "every operation has its operands in the intended order",
        "an operation has its two operands swapped",
    ),
    "code_wrong_operator": (
        "wrong_binary_operator_datasets",
        "Wrong binary operator",
        "Does this function use a wrong binary operator somewhere?",
        [
            "Is one of the operators in this code the wrong one?",
            "Does this code contain a wrong-operator bug?",
            "Has a binary operator been replaced by an incorrect one?",
            "Is there an arithmetic, comparison or boolean operator here that should be different?",
        ],
        "every operator is the intended one",
        "an operator is the wrong one",
    ),
}
REVIEW_TASK = "code_review_comment"
REVIEW_Q = (
    "Would a code reviewer leave a comment on this change?",
    [
        "Does this diff need a reviewer's comment before it is merged?",
        "Would this code change draw a review comment?",
        "Is this change one a reviewer would remark on?",
        "Would a careful reviewer ask for something in this diff?",
    ],
    "the change would be accepted without a comment",
    "a reviewer would comment on the change",
)
EXCEPTION_TASK = "code_exception_type"
EXCEPTION_Q = (
    "Which exception does the handler marked __HOLE__ catch?",
    [
        "What exception type belongs in place of __HOLE__?",
        "Which exception class does the `except __HOLE__` clause handle?",
        "What should __HOLE__ be, given what the try block can raise?",
        "Which error is the masked except clause written to catch?",
    ],
)
EXCEPTION_OPTIONS = 5  # the gold class and four others


# The variant suffixes CuBERT appends to a function's `info` path: `/original`, and one per
# mutation (`/VarMisuse@26/32 `self`->`args``, `/swapped operands of `%``, `/operator `==` -> `!=``).
VARIANT = re.compile(r"/(?:original|VarMisuse@.*|swapped operands of .*|operator .*)$")


def _function_key(info: str) -> str:
    """The function a CuBERT row comes from: its `info` path without the variant suffix."""
    return VARIANT.sub("", info)


def _read_parquet(path: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def _balanced(rows: list[tuple[str, int]], n: int, rng: random.Random) -> list[tuple[str, int]]:
    """n rows, half with label 1 and half with label 0 (fewer if a class runs short)."""
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def cubert_noul(path: str, task: str, n: int, max_chars: int, rng: random.Random) -> list[Example]:
    config, yes, question, variants, false_text, true_text = CUBERT_NOUL[task]
    del config
    seen: set[str] = set()
    rows: list[tuple[str, int]] = []
    records = _read_parquet(path)
    rng.shuffle(records)
    for r in records:
        key = _function_key(str(r.get("info", "")))
        if key in seen:
            continue
        code = str(r.get("function") or "").strip("\n")
        if task == "code_docstring_match":
            doc = str(r.get("docstring") or "").strip()
            if not doc:
                continue
            state = f"Function:\n{code}\n\nDocstring:\n{doc}"
        else:
            state = f"Python function:\n{code}"
        if not code.strip() or len(state) > max_chars:
            continue
        seen.add(key)
        rows.append((state, int(r["label"] == yes)))
    return [
        Example(
            kind="noul",
            state=state,
            instructions=question,
            options=[["false", false_text], ["true", true_text]],
            label=label,
            task=task,
            instruction_variants=list(variants),
        )
        for state, label in _balanced(rows, n, rng)
    ]


def cubert_exception(path: str, n: int, max_chars: int, rng: random.Random) -> list[Example]:
    records = _read_parquet(path)
    classes = sorted({str(r["label"]) for r in records})
    rng.shuffle(records)
    out: list[Example] = []
    seen: set[str] = set()
    for r in records:
        if len(out) >= n:
            break
        code = str(r.get("function") or "").strip("\n")
        key = _function_key(str(r.get("info", "")))
        if key in seen or "__HOLE__" not in code:
            continue
        state = f"Python function (one exception name is masked as __HOLE__):\n{code}"
        if len(state) > max_chars:
            continue
        seen.add(key)
        gold = str(r["label"])
        others = rng.sample([c for c in classes if c != gold], EXCEPTION_OPTIONS - 1)
        names = sorted([gold, *others])
        out.append(
            Example(
                kind="choice",
                state=state,
                instructions=EXCEPTION_Q[0],
                options=[[c, f"the handler catches {c}"] for c in names],
                label=names.index(gold),
                task=EXCEPTION_TASK,
                instruction_variants=list(EXCEPTION_Q[1]),
            )
        )
    return out


def _review_rows(path: str, max_chars: int) -> Iterator[tuple[str, int]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            patch = str(r.get("patch") or "").strip("\n")
            if not patch:
                continue
            state = f"Language: {r.get('lang') or 'unknown'}\n\nDiff:\n{patch}"
            if len(state) <= max_chars:
                yield state, int(str(r.get("y")) == "1")


def codereviewer(path: str, n: int, max_chars: int, rng: random.Random) -> list[Example]:
    question, variants, false_text, true_text = REVIEW_Q
    return [
        Example(
            kind="noul",
            state=state,
            instructions=question,
            options=[["false", false_text], ["true", true_text]],
            label=label,
            task=REVIEW_TASK,
            instruction_variants=list(variants),
        )
        for state, label in _balanced(list(_review_rows(path, max_chars)), n, rng)
    ]


def build(
    cubert: str, review: str, per_task: int, max_chars: int, seed: int, split: str
) -> list[Example]:
    """The rows of one split: `train` (train/validation files) or `test` (test files)."""
    rng = random.Random(f"{seed}:{split}")
    rows: list[Example] = []
    for task, (config, *_rest) in CUBERT_NOUL.items():
        path = os.path.join(cubert, config, f"{split}-00000-of-00001.parquet")
        rows += cubert_noul(path, task, per_task, max_chars, rng)
    rows += cubert_exception(
        os.path.join(cubert, "exception_datasets", f"{split}-00000-of-00001.parquet"),
        per_task,
        max_chars,
        rng,
    )
    review_file = "cls-valid.jsonl" if split == "train" else "cls-test.jsonl"
    rows += codereviewer(os.path.join(review, "quality", review_file), per_task, max_chars, rng)
    rng.shuffle(rows)
    return rows


def _counts(rows: Iterable[Example]) -> str:
    from collections import Counter

    c = Counter((r.task, r.label if r.kind == "noul" else "choice") for r in rows)
    return ", ".join(f"{t}:{lab}={n}" for (t, lab), n in sorted(c.items(), key=str))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cubert", required=True, help="local copy of claudios/cubert_ETHPy150Open")
    p.add_argument(
        "--codereviewer", required=True, help="local copy of fasterinnerlooper/codereviewer"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--eval-out", required=True)
    p.add_argument("--per-task", type=int, default=2000, help="training rows per task")
    p.add_argument("--eval-per-task", type=int, default=200, help="held-out rows per task")
    p.add_argument("--max-chars", type=int, default=6000, help="drop states longer than this")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    train = build(a.cubert, a.codereviewer, a.per_task, a.max_chars, a.seed, "train")
    held = build(a.cubert, a.codereviewer, a.eval_per_task, a.max_chars, a.seed, "test")
    # The sources' splits come from different repositories, but a short function can occur in
    # both. A held-out row must be unseen, so a state that training holds leaves the held-out set.
    seen = {r.state for r in train}
    dropped = sum(r.state in seen for r in held)
    held = [r for r in held if r.state not in seen]
    if dropped:
        print(f"{a.eval_out}: {dropped} held-out rows dropped, their states are in training")
    print(f"{a.out}: {write_jsonl(a.out, train)} rows ({_counts(train)})")
    print(f"{a.eval_out}: {write_jsonl(a.eval_out, held)} rows ({_counts(held)})")


if __name__ == "__main__":
    main()
