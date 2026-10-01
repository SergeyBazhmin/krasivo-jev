import hashlib
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from datasets import load_from_disk
from loguru import logger

from jev_model.data import content_key, partition
from jev_model.pointer.encode import Encoder, Tokens

# what stands in for the true option when a draw hides it
NONE_TEXT = "None of the above"


@dataclass
class Item:
    tokens: Tokens
    label: list[float]
    # what the option shuffles are seeded by
    key: str
    length: int


@dataclass
class Draw:
    """One training draw: a sample and the order its options are shown in."""

    item: Item
    order: list[int]


def load_items(
    names: list[str], data_dir: Path, part: str, encoder: Encoder, limit: int = 0
) -> dict[str, list[Item]]:
    """{dataset: items} for the datasets that have rows in `part`. `limit` keeps that many rows per
    dataset, chosen by a hash of the content, so every run scores the same ones."""
    loaded = {}
    for name in names:
        samples = partition(load_from_disk(data_dir / name))[part]
        if limit and len(samples) > limit:
            samples = sorted(samples, key=lambda s: hashlib.sha1(sample_key(s).encode()).digest())[:limit]
        tokenized = encoder.tokenize([(s["state"], s["question"], [o["text"] for o in s["options"]]) for s in samples])
        items = [
            Item(tokens, sample["label"], sample_key(sample), encoder.length(tokens))
            for sample, tokens in zip(samples, tokenized)
            if len(tokens.options) >= 2 and encoder.fits(tokens)
        ]
        logger.info(f"[{name}] {part}: {len(items)} samples, {len(samples) - len(items)} skipped")
        if items:
            loaded[name] = items
    return loaded


def sample_key(sample: dict) -> str:
    return f"{content_key(sample)}:{sample['question']}"


def token_batches[T](items: list[T], lengths: list[int], batch_tokens: int) -> Iterator[list[T]]:
    """`items` grouped by similar length, each batch at most `batch_tokens` once padded."""
    batch: list[T] = []
    for i in sorted(range(len(items)), key=lambda i: lengths[i]):
        if batch and lengths[i] * (len(batch) + 1) > batch_tokens:
            yield batch
            batch = []
        batch.append(items[i])
    if batch:
        yield batch


class Pool:
    """Every training sample, drawn with per-dataset mixing weights and a fresh option order per draw,
    so no position can learn a prior for an answer. With probability `none_prob` a draw shows
    `NONE_TEXT` in place of the true option, so the model does not learn that a listed answer must be right."""

    # draws sorted by length together before they are cut into batches; a larger block pads less
    BLOCK = 2048

    def __init__(self, data: dict[str, list[Item]], alpha: float, seed: int, encoder: Encoder, none_prob: float = 0.0):
        self.items = [item for items in data.values() for item in items]
        self.weights = torch.cat([torch.full((len(items),), len(items) ** alpha / len(items), dtype=torch.float64)
                                  for items in data.values()])
        self.seed = seed
        self.encoder = encoder
        self.none_prob = none_prob
        self.none = encoder.tokenize([("", "", [NONE_TEXT])])[0].options[0]

    def draw(self, index: int, count: int) -> Draw:
        item = self.items[index]
        order = list(range(len(item.tokens.options)))
        rng = random.Random(f"{self.seed}:{count}:{item.key}")
        rng.shuffle(order)
        if rng.random() < self.none_prob:
            item = self.without_answer(item)
        return Draw(item, order)

    def without_answer(self, item: Item) -> Item:
        """`item` with its true option replaced by the none option, which takes over the label. Left as it
        is when there is no single true option (soft labels), when the source already has a none option,
        or when the longer sequence would not fit."""
        if max(item.label) < 1.0 or self.none in item.tokens.options:
            return item
        options = list(item.tokens.options)
        options[item.label.index(max(item.label))] = self.none
        tokens = replace(item.tokens, options=options)
        if not self.encoder.fits(tokens):
            return item
        return replace(item, tokens=tokens, length=self.encoder.length(tokens))

    def batches(self, batch_tokens: int, generator: torch.Generator) -> Iterator[list[Draw]]:
        count = 0
        while True:
            indices = torch.multinomial(self.weights, self.BLOCK, replacement=True, generator=generator).tolist()
            draws = [self.draw(index, count + i) for i, index in enumerate(indices)]
            count += len(draws)
            batches = list(token_batches(draws, [d.item.length for d in draws], batch_tokens))
            for i in torch.randperm(len(batches), generator=generator).tolist():
                yield batches[i]


class SizedEpochPool(Pool):
    """A pool for training by epochs: every pass shuffles the whole training set and cuts it into batches of
    `batch_size` draws in that order, so each sample is shown once per pass. Datasets are mixed by size (`alpha`
    has no effect), and the batches are cut by count rather than padded tokens, so their padded size varies with
    the lengths of their draws."""

    def __init__(self, *args, batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size < 1:
            raise ValueError("sized sampling needs batch_size >= 1")
        self.batch_size = batch_size

    def epoch(self, number: int, generator: torch.Generator) -> Iterator[list[Draw]]:
        """The batches of one pass, `batch_size` draws each (the last may hold fewer)."""
        order = torch.randperm(len(self.items), generator=generator).tolist()
        draws = [self.draw(index, number * len(self.items) + position) for position, index in enumerate(order)]
        for start in range(0, len(draws), self.batch_size):
            yield draws[start:start + self.batch_size]

    def batches(self, batch_tokens: int, generator: torch.Generator) -> Iterator[list[Draw]]:
        """Batches of `batch_size` draws, pass after pass; `batch_tokens` is ignored."""
        number = 0
        while True:
            yield from self.epoch(number, generator)
            number += 1

    def count_batches(self, epochs: int) -> int:
        """How many batches the first `epochs` passes of `batches` hold."""
        return epochs * math.ceil(len(self.items) / self.batch_size)


POOLS: dict[str, type[Pool]] = {"weighted": Pool, "sized": SizedEpochPool}
