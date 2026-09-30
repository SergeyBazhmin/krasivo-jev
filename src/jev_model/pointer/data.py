import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk
from loguru import logger

from jev_model.data import content_key, partition
from jev_model.pointer.encode import Encoder, Tokens


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
    so no position can learn a prior for an answer."""

    # draws sorted by length together before they are cut into batches; a larger block pads less
    BLOCK = 2048

    def __init__(self, data: dict[str, list[Item]], alpha: float, seed: int):
        self.items = [item for items in data.values() for item in items]
        self.weights = torch.cat([torch.full((len(items),), len(items) ** alpha / len(items), dtype=torch.float64)
                                  for items in data.values()])
        self.seed = seed

    def draw(self, index: int, count: int) -> Draw:
        item = self.items[index]
        order = list(range(len(item.tokens.options)))
        random.Random(f"{self.seed}:{count}:{item.key}").shuffle(order)
        return Draw(item, order)

    def batches(self, batch_tokens: int, generator: torch.Generator) -> Iterator[list[Draw]]:
        count = 0
        while True:
            indices = torch.multinomial(self.weights, self.BLOCK, replacement=True, generator=generator).tolist()
            draws = [self.draw(index, count + i) for i, index in enumerate(indices)]
            count += len(draws)
            batches = list(token_batches(draws, [d.item.length for d in draws], batch_tokens))
            for i in torch.randperm(len(batches), generator=generator).tolist():
                yield batches[i]
