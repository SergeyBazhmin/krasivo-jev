"""Runs `disjoint_contexts` (what `JevDataset.regroup` calls) on tiny hand-made splits and prints what happens.

    uv run python src/scripts/debug_regroup.py

Each case names a document and the splits its rows start in; compare "before" and "after" to see where every row ends
up. Run it once as is (rows carry a source `group`) and once with `--no-group` (grouping by `state` only) to see which
leaks only the `group` column catches.
"""

import sys

from datasets import Dataset, DatasetDict

from jev_datasets.utils import (
    GROUPED_FEATURES,
    SAMPLE_FEATURES,
    disjoint_contexts,
    group_key,
    make_options,
    make_sample,
)

OPTIONS = make_options(["yes", "no"])


def row(doc: str, state: str, question: str, answer: str = "yes") -> dict:
    # `doc` stands in for `content_group(...)` of the source document, which `map_records` copies onto each sample.
    return {**make_sample(state, question, OPTIONS, answer), "group": doc}


SPLITS = {
    "train": [
        # A: a contract cut into passages; the train copy yields a passage the test copy does not.
        row("contract-A", "A, passage about expiration", "Does it expire?"),
        # B: the same passage and question as in test, so an exact duplicate.
        row("contract-B", "B, passage about law", "Is the law Delaware?"),
        # C: same state as in validation, different question; moves by state alone.
        row("contract-C", "C, passage about fees", "Is there a fee?"),
        # D: only in train, two passages; stays put.
        row("contract-D", "D, passage one", "Q1?"),
        row("contract-D", "D, passage two", "Q2?", "no"),
    ],
    "validation": [
        row("contract-C", "C, passage about fees", "Is the fee monthly?", "no"),
        # E: the same state twice within one split; an exact duplicate inside a split.
        row("contract-E", "E, passage", "Q?"),
        row("contract-E", "E, passage", "Q?"),
    ],
    "test": [
        row("contract-A", "A, passage about termination", "Can it be terminated?"),
        row("contract-B", "B, passage about law", "Is the law Delaware?"),
    ],
}


def show(title: str, data: DatasetDict) -> None:
    print(f"\n== {title}")
    for split, rows in data.items():
        for sample in rows:
            doc = sample.get("group", "-")
            print(
                f"  {split:<10} {doc:<11} {sample['state']:<32} {sample['question']:<24} key={group_key(sample)[:12]}"
            )


def main() -> None:
    use_group = "--no-group" not in sys.argv
    features = GROUPED_FEATURES if use_group else SAMPLE_FEATURES
    data = DatasetDict(
        {
            split: Dataset.from_list([{k: v for k, v in r.items() if k in features} for r in rows], features=features)
            for split, rows in SPLITS.items()
        }
    )
    show(f"before (grouping by {'group' if use_group else 'state'})", data)
    regrouped, moved, dropped = disjoint_contexts(data)
    show("after", regrouped)
    print("\nmoved:  ", dict(moved))
    print("dropped:", dict(dropped))
    print("columns:", next(iter(regrouped.values())).column_names)
    states = {split: set(rows["state"]) for split, rows in regrouped.items()}
    for a, first in states.items():
        for b, second in states.items():
            if a < b and (shared := first & second):
                print(f"LEAK {a}/{b}: {sorted(shared)}")
    # The saved rows have no `group`; recover each one's document from the input to check document-level leaks.
    doc_of = {r["state"]: r["group"] for rows in SPLITS.values() for r in rows}
    by_doc: dict[str, set[str]] = {}
    for split, rows in regrouped.items():
        for state in rows["state"]:
            by_doc.setdefault(doc_of[state], set()).add(split)
    for doc, splits in sorted(by_doc.items()):
        if len(splits) > 1:
            print(f"DOCUMENT LEAK {doc}: rows in {sorted(splits)}")


if __name__ == "__main__":
    main()
