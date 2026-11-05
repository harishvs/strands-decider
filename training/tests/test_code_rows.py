"""training.data.code_rows on tiny CuBERT- and CodeReviewer-shaped files."""

from __future__ import annotations

import json
import os
from collections import Counter

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from training.data import code_rows as C  # noqa: E402
from training.data.format import read_jsonl  # noqa: E402

N_FUNC = 40  # functions per split; each has an original and a mutated copy


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _sources(root):
    cubert, review = root / "cubert", root / "codereviewer"
    for split in ("train", "test"):
        tag = f"{split}{'' if split == 'train' else '_t'}"
        for task, (config, yes, *_rest) in C.CUBERT_NOUL.items():
            rows = []
            for i in range(N_FUNC):
                base = f"dataset/ETHPy150Open org/repo/{tag}/m{i}.py/f{i}"
                if task == "code_docstring_match":
                    rows.append(
                        {
                            "function": f"def f{i}():\n    return {i}  # {tag}",
                            "docstring": f"Return {i}.",
                            "label": "Correct",
                            "info": base,
                        }
                    )
                    rows.append(
                        {
                            "function": f"def f{i}():\n    return {i}  # {tag}",
                            "docstring": "Open a file.",
                            "label": "Incorrect",
                            "info": base,
                        }
                    )
                else:
                    suffix = {
                        "code_variable_misuse": "VarMisuse@3/7 `a`->`b`",
                        "code_swapped_operands": "swapped operands of `-`",
                        "code_wrong_operator": "operator `+` -> `-`",
                    }[task]
                    rows.append(
                        {
                            "function": f"def f{i}(a, b):\n    return a + b + {i}  # {tag}",
                            "label": "Correct",
                            "info": f"{base}/original",
                        }
                    )
                    rows.append(
                        {
                            "function": f"def f{i}(a, b):\n    return b - a + {i}  # {tag}",
                            "label": yes,
                            "info": f"{base}/{suffix}",
                        }
                    )
            _write(str(cubert / config / f"{split}-00000-of-00001.parquet"), rows)
        exc = ["KeyError", "ValueError", "OSError", "TypeError", "IndexError", "AttributeError"]
        _write(
            str(cubert / "exception_datasets" / f"{split}-00000-of-00001.parquet"),
            [
                {
                    "function": f"def g{i}():\n    try:\n        x[{i}]  # {tag}\n    except __HOLE__:\n        pass",
                    "label": exc[i % len(exc)],
                    "info": f"dataset/ETHPy150Open org/repo/{tag}/e{i}.py/g{i}",
                }
                for i in range(N_FUNC)
            ],
        )
        os.makedirs(review / "quality", exist_ok=True)
        name = "cls-valid.jsonl" if split == "train" else "cls-test.jsonl"
        with open(review / "quality" / name, "w") as fh:
            for i in range(N_FUNC):
                fh.write(
                    json.dumps(
                        {
                            "patch": f"@@ -1 +1 @@\n-x = {i}\n+x = {i} # {tag}",
                            "y": str(i % 2),
                            "oldf": "x" * 50,
                            "lang": "py",
                        }
                    )
                    + "\n"
                )
    return str(cubert), str(review)


def test_rows_are_balanced_one_per_function_and_held_out(tmp_path):
    cubert, review = _sources(tmp_path)
    out, held = tmp_path / "code.jsonl", tmp_path / "code_eval.jsonl"
    C.main(
        [
            "--cubert",
            cubert,
            "--codereviewer",
            review,
            "--out",
            str(out),
            "--eval-out",
            str(held),
            "--per-task",
            "20",
            "--eval-per-task",
            "10",
        ]
    )
    train, test = list(read_jsonl(str(out))), list(read_jsonl(str(held)))
    tasks = Counter(r.task for r in train)
    assert set(tasks) == {*C.CUBERT_NOUL, C.REVIEW_TASK, C.EXCEPTION_TASK}
    for task in [*C.CUBERT_NOUL, C.REVIEW_TASK]:
        labels = Counter(r.label for r in train if r.task == task)
        assert labels[0] == labels[1] == 10, (task, labels)  # balanced to per-task
    for r in train:
        if r.kind == "noul":
            assert [o[0] for o in r.options] == ["false", "true"] and len(
                r.instruction_variants
            ) == 4
    # one row per function: never a function's original and its mutated copy both
    names = [
        r.state.split("def ")[1].split("(")[0] for r in train if r.task == "code_variable_misuse"
    ]
    assert len(names) == len(set(names))
    exc = [r for r in train if r.task == C.EXCEPTION_TASK]
    assert exc and all(r.kind == "choice" and r.n_options == C.EXCEPTION_OPTIONS for r in exc)
    assert all("__HOLE__" in r.state for r in exc)
    assert {r.state for r in train}.isdisjoint({r.state for r in test})  # test files only


def test_a_rebuild_is_identical(tmp_path):
    cubert, review = _sources(tmp_path)
    outs = []
    for k in ("a", "b"):
        out = tmp_path / f"{k}.jsonl"
        C.main(
            [
                "--cubert",
                cubert,
                "--codereviewer",
                review,
                "--out",
                str(out),
                "--eval-out",
                str(tmp_path / f"{k}_eval.jsonl"),
                "--per-task",
                "20",
                "--eval-per-task",
                "10",
            ]
        )
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


def test_long_states_are_dropped(tmp_path):
    cubert, review = _sources(tmp_path)
    out = tmp_path / "code.jsonl"
    C.main(
        [
            "--cubert",
            cubert,
            "--codereviewer",
            review,
            "--out",
            str(out),
            "--eval-out",
            str(tmp_path / "e.jsonl"),
            "--per-task",
            "20",
            "--eval-per-task",
            "10",
            "--max-chars",
            "60",
        ]
    )
    assert all(len(r.state) <= 60 for r in read_jsonl(str(out)))


def test_function_keys_strip_every_cubert_variant_suffix():
    base = "dataset/ETHPy150Open org/repo/a.py Cls.meth"
    for suffix in (
        "original",
        "VarMisuse@26/32 `self`->`args`",
        "swapped operands of `/`",
        "operator `==` -> `!=`",
    ):
        assert C._function_key(f"{base}/{suffix}") == base
    assert (
        C._function_key("dataset/ETHPy150Open org/repo/a.py/f")
        == "dataset/ETHPy150Open org/repo/a.py/f"
    )


def test_a_held_out_row_seen_in_training_is_dropped(tmp_path):
    cubert, review = _sources(tmp_path)
    # the test split of one task repeats a training function verbatim
    path = os.path.join(cubert, "swapped_operands_datasets", "test-00000-of-00001.parquet")
    rows = pq.read_table(path).to_pylist()
    train_path = os.path.join(cubert, "swapped_operands_datasets", "train-00000-of-00001.parquet")
    train_rows = pq.read_table(train_path).to_pylist()
    _write(path, train_rows + rows)
    out, held = tmp_path / "code.jsonl", tmp_path / "code_eval.jsonl"
    C.main(
        [
            "--cubert",
            cubert,
            "--codereviewer",
            review,
            "--out",
            str(out),
            "--eval-out",
            str(held),
            "--per-task",
            "20",
            "--eval-per-task",
            "10",
        ]
    )
    train, test = list(read_jsonl(str(out))), list(read_jsonl(str(held)))
    assert {r.state for r in train}.isdisjoint({r.state for r in test})
