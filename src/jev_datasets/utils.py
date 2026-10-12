import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets

# Pinned explicitly: without it `map` inherits a source column's type for same-named
# output columns (e.g. a ClassLabel `label` rejects what we write into it).
SAMPLE_FEATURES = Features(
    {
        "state": Value("string"),
        "question": Value("string"),
        "options": [{"id": Value("string"), "text": Value("string")}],
        "label": [Value("float32")],
    }
)
# What converters of grouped sources emit: `group` keys the source document for `regroup`, which removes it.
GROUPED_FEATURES = Features({**SAMPLE_FEATURES, "group": Value("string")})


def slugify(text: str) -> str:
    """Stable machine key for an option that has no natural id of its own."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def stable_shuffle(items: list, seed: str) -> list:
    """Shuffled the same way on every run; for sources that always put the right answer first."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def pick_question(versions: list[str], seed: str) -> str:
    """One phrasing of the question per sample, the same on every run; salted so it doesn't
    move in step with a `stable_shuffle` of the same seed and leak the answer's position."""
    return random.Random(f"question:{seed}").choice(versions)


def make_option(id: str, text: str) -> dict[str, str]:
    return {"id": id, "text": text}


def make_options(ids: list[str], texts: list[str] | None = None) -> list[dict[str, str]]:
    """Answer candidates; `ids` are what a single-answer label refers to, `texts` what the model reads."""
    return [make_option(id, text) for id, text in zip(ids, texts if texts is not None else ids)]


def as_distribution(options: list[dict[str, str]], label: str | list[float]) -> list[float]:
    """A soft target as given, or a one-hot over `options` for a single correct answer id."""
    if isinstance(label, str):
        if not any(option["id"] == label for option in options):
            raise ValueError(f"label {label!r} is not one of {[o['id'] for o in options]}")
        return [float(option["id"] == label) for option in options]
    if len(label) != len(options):
        raise ValueError(f"got {len(label)} probabilities for {len(options)} options")
    return [float(p) for p in label]


def with_context(about: str, text: str, name: str | None = None, answers: dict[str, str] | None = None) -> str:
    """`text` under a system-prompt-like instruction (`about`: who the model is, what it is given, what to
    decide), plus what each answer means when `answers` is given. The model stores no facts, so a source's
    conventions have to be in the state."""
    guide = "\n".join(f"- {answer}: {meaning}" for answer, meaning in (answers or {}).items())
    parts = (about, guide and f"Answers:\n{guide}", f"{name}:\n{text}" if name else text)
    return "\n\n".join(part for part in parts if part)


def make_sample(state: str, question: str, options: list[dict[str, str]], label: str | list[float]):
    """
    state:    the context the question is asked about ("" when there is none)
    question: the question itself
    options:  the answer candidates, as {"id": ..., "text": ...}
    label:    a probability distribution over `options`, or the `id` of the single
              correct option, which becomes 1.0 with 0.0 everywhere else
    """
    return {"state": state, "question": question, "options": options, "label": as_distribution(options, label)}


def text_choice(state: str, question: str, answer: str, distractors: list[str], seed: str) -> dict | None:
    """Distinct text candidates with shuffled positional ids, or no sample if there is no distractor."""
    texts = [answer]
    seen = {answer.strip().casefold()}
    for text in stable_shuffle(distractors, f"distractors:{seed}"):
        key = text.strip().casefold()
        if key and key not in seen:
            texts.append(text)
            seen.add(key)
        if len(texts) == 4:
            break
    if len(texts) < 2:
        return None
    texts = stable_shuffle(texts, f"options:{seed}")
    ids = list("ABCD")[: len(texts)]
    return make_sample(state, question, make_options(ids, texts), ids[texts.index(answer)])


def explode(fn: Callable[[dict], list[dict]], features: Features = SAMPLE_FEATURES) -> Callable[[dict], dict]:
    """Adapts `fn(row) -> [sample, ...]` to a batched `map`, for sources with several samples per row."""

    def batched(batch: dict[str, list]) -> dict[str, list]:
        samples = [sample for row in zip(*batch.values()) for sample in fn(dict(zip(batch, row)))]
        return {column: [sample[column] for sample in samples] for column in features}

    return batched


def group_key(sample: dict) -> str:
    """Rows about one context (several questions on one passage) form a group that must stay in one split.
    A source `group` wins, as one document can yield several different states."""
    return sample.get("group") or hashlib.sha1((sample["state"] or sample["question"]).encode()).hexdigest()


def row_key(sample: dict) -> str:
    content = [sample["state"], sample["question"], sample["options"], sample["label"]]
    return hashlib.sha1(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def disjoint_contexts(data: DatasetDict) -> tuple[DatasetDict, Counter, Counter]:
    """Every group in exactly one split: the first one it appears in, held-out splits first. Rows from other splits
    are moved there; only exact duplicate rows are dropped, keeping the held-out copy. Also returns how many rows
    moved, by (from, to), and how many duplicates were dropped, by split of origin. Removes the `group` column."""
    priority = {"test": 0, "validation": 1, "dev": 1, "val": 1, "train": 2}
    order = sorted(data, key=lambda split: (priority.get(split, 2), split))
    columns = [
        column for column in ("state", "question", "options", "label", "group") if column in data[order[0]].column_names
    ]
    rows = {split: list(data[split].select_columns(columns)) for split in data}
    keys = {split: [group_key(row) for row in rows[split]] for split in data}
    home: dict[str, str] = {}
    for split in order:
        for key in keys[split]:
            home.setdefault(key, split)
    moved = Counter((split, home[key]) for split in data for key in keys[split] if home[key] != split)
    dropped: Counter = Counter()
    seen: set[str] = set()
    selected: dict[str, list[list[int]]] = {target: [] for target in data}
    for split in order:
        picks = {target: [] for target in data}
        for i, (key, row) in enumerate(zip(keys[split], rows[split])):
            if (identity := row_key(row)) in seen:
                dropped[split] += 1
                continue
            seen.add(identity)
            picks[home[key]].append(i)
        for target in data:
            selected[target].append(picks[target])
    regrouped = {
        target: concatenate_datasets([data[split].select(indices) for split, indices in zip(order, selected[target])])
        for target in data
    }
    if "group" in columns:
        regrouped = {split: rows.remove_columns("group") for split, rows in regrouped.items()}
    # a split whose every group lives elsewhere ends up empty; drop it so `partition` carves it instead
    return DatasetDict({split: rows for split, rows in regrouped.items() if rows.num_rows}), moved, dropped


def stratified_limit(dataset: Dataset, max_samples: int, seed: int = 0) -> Dataset:
    """At most `max_samples` rows, drawn so each answer (argmax of `label`) keeps its share."""
    if dataset.num_rows <= max_samples:
        return dataset
    strata = np.array([int(np.argmax(label)) for label in dataset["label"]])
    _, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    # largest-remainder rounding, so the quotas add up to exactly `max_samples`
    quotas = counts * max_samples / dataset.num_rows
    take = np.floor(quotas).astype(int)
    take[np.argsort(take - quotas, kind="stable")[: max_samples - take.sum()]] += 1
    rng = np.random.default_rng(seed)
    indices = np.concatenate(
        [rng.choice(np.flatnonzero(inverse == k), size=n, replace=False) for k, n in enumerate(take)]
    )
    return dataset.select(np.sort(indices))
