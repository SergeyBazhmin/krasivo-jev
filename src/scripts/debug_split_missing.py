"""Toy DatasetDicts run through `split_missing`, with no Hub access, for stepping through it in a debugger.

    uv run python src/scripts/debug_split_missing.py [case]

Put a breakpoint in `jev_datasets.utils.split_missing` (or on `targets = ...`) and run one case.
"""

import sys

from datasets import Dataset, DatasetDict

from jev_datasets.utils import (
    GROUPED_FEATURES,
    SAMPLE_FEATURES,
    canonical_splits,
    disjoint_contexts,
    split_missing,
)

OPTIONS = [{"id": "yes", "text": "yes"}, {"id": "no", "text": "no"}]


def rows(prefix: str, n: int, group_size: int = 0) -> Dataset:
    """`n` yes/no samples; with `group_size`, every `group_size` consecutive rows share a document `group` but have
    different states, the case where only the group keeps them together."""
    samples = [
        {
            "state": f"{prefix} passage {i}",
            "question": "Is it true?",
            "options": OPTIONS,
            "label": [1.0, 0.0] if i % 2 else [0.0, 1.0],
            **({"group": f"{prefix} doc {i // group_size}"} if group_size else {}),
        }
        for i in range(n)
    ]
    return Dataset.from_list(samples, features=GROUPED_FEATURES if group_size else SAMPLE_FEATURES)


CASES = {
    # train exists, nothing held out: 5% of train to test, 5% to validation
    "train_only": lambda: DatasetDict({"train": rows("t", 400)}),
    # train + test: only validation is carved; the source test is untouched
    "train_test": lambda: DatasetDict({"train": rows("t", 400), "test": rows("x", 50)}),
    # halueval-style unknown name and a `dev` split: renamed to train and validation, only test carved
    "aliases": lambda: DatasetDict({"data": rows("d", 400), "dev": rows("v", 40)}),
    # eval-only source (rublimp): every row re-split 80/10/10
    "test_only": lambda: DatasetDict({"test": rows("x", 400)}),
    # eval-only source with validation + test: both pooled and re-split
    "val_test": lambda: DatasetDict({"validation": rows("v", 200), "test": rows("x", 200)}),
    # documents of 4 passages each: a document must not straddle splits
    "grouped": lambda: DatasetDict({"train": rows("g", 400, group_size=4)}),
    # too small to fill every bucket range: expect an empty split
    "tiny": lambda: DatasetDict({"test": rows("x", 5)}),
}


def run(name: str):
    data = CASES[name]()
    print(f"\n=== {name}: in {dict(data.num_rows)}")
    # what `JevDataset.regroup` does before `split_missing`, which expects canonical names and whole groups
    data, _, _ = disjoint_contexts(canonical_splits(data))
    grouped = "group" in next(iter(data.values())).column_names
    out, carved = split_missing(data)
    print(f"out {dict(out.num_rows)}, carved {dict(carved)}")

    assert list(out) == ["train", "validation", "test"], list(out)
    assert "group" not in out["train"].column_names
    keys = {split: {row["state"] or row["question"] for row in part} for split, part in out.items()}
    assert not keys["train"] & keys["test"] and not keys["train"] & keys["validation"]
    assert not keys["validation"] & keys["test"]
    if grouped:
        # each document's rows all share its bucket, so the whole document must sit in one split
        where = {}
        for split, part in out.items():
            for row in part:
                doc = row["state"].rsplit(" ", 1)[1]
                where.setdefault(int(doc) // 4, set()).add(split)
        assert all(len(splits) == 1 for splits in where.values()), "a document straddles splits"
        print(f"{len(where)} documents, each in one split")
    if empty := [split for split, part in out.items() if not part.num_rows]:
        print(f"empty: {empty}")
    print("first test row:", out["test"][0] if out["test"].num_rows else None)


if __name__ == "__main__":
    for name in sys.argv[1:] or CASES:
        run(name)
