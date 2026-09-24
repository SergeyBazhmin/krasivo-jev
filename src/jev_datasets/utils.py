import random
import re
from collections.abc import Callable

from datasets import Features, Value

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


def slugify(text: str) -> str:
    """Stable machine key for an option that has no natural id of its own."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def stable_shuffle(items: list, seed: str) -> list:
    """Shuffled the same way on every run; for sources that always put the right answer first."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


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


def make_sample(state: str, question: str, options: list[dict[str, str]], label: str | list[float]):
    """
    state:    the context the question is asked about ("" when there is none)
    question: the question itself
    options:  the answer candidates, as {"id": ..., "text": ...}
    label:    a probability distribution over `options`, or the `id` of the single
              correct option, which becomes 1.0 with 0.0 everywhere else
    """
    return {
        "state": state,
        "question": question,
        "options": options,
        "label": as_distribution(options, label)
    }


def explode(fn: Callable[[dict], list[dict]]) -> Callable[[dict], dict]:
    """Adapts `fn(row) -> [sample, ...]` to a batched `map`, for sources with several samples per row."""

    def batched(batch: dict[str, list]) -> dict[str, list]:
        samples = [sample for row in zip(*batch.values()) for sample in fn(dict(zip(batch, row)))]
        return {column: [sample[column] for sample in samples] for column in SAMPLE_FEATURES}

    return batched
