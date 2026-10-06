"""Raw records keep source groups intact before any one-to-many conversion."""

import hashlib
import json
from collections.abc import Callable, Iterable

from datasets import Dataset, DatasetDict


def record(payload: dict, group: str) -> dict[str, str]:
    return {"group": group, "payload": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def content_group(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def grouped_records(generate: Callable[..., Iterable[dict]], **kwargs) -> DatasetDict:
    """Holdouts win duplicate source groups; each group has exactly one split even after conversion."""
    rows = Dataset.from_generator(generate, gen_kwargs=kwargs)
    priority = {"test": 0, "validation": 1, "train": 2}
    owners = {}
    for group, split in zip(rows["group"], rows["split"]):
        if group not in owners or priority[split] < priority[owners[group]]:
            owners[group] = split
    indices = {split: [] for split in ("train", "validation", "test")}
    seen = set()
    for index, row in enumerate(rows):
        if owners[row["group"]] != row["split"]:
            continue
        identity = row["group"], content_group(row["payload"])
        if identity not in seen:
            indices[row["split"]].append(index)
            seen.add(identity)
    return DatasetDict({split: rows.select(selected) for split, selected in indices.items()})
