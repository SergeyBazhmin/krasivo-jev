import hashlib
from pathlib import Path

from datasets import DatasetDict, load_from_disk

from jev_model.constants import PARTITIONS

SPLIT_ALIASES = {"train": "train", "validation": "validation", "dev": "validation", "val": "validation", "test": "test"}
# share of the train pool carved off when a source has no split of its own for it
HOLDOUT_PERCENT = 5


def built_names(data_dir: Path) -> list[str]:
    """Datasets that `jev prepare` has saved under `data_dir`."""
    return sorted(p.parent.name for p in data_dir.glob("*/dataset_dict.json"))


def resolve_names(names: list[str], data_dir: Path) -> list[str]:
    """`names`, or every built dataset when it is empty; fails on names that were never built."""
    built = built_names(data_dir)
    if missing := [name for name in names if name not in built]:
        raise FileNotFoundError(f"not built under {data_dir}: {', '.join(missing)} (run `jev prepare`)")
    return names or built


def content_key(sample: dict) -> str:
    # rows sharing a state (several questions on one passage) must land in the same partition
    return sample["state"] or sample["question"]


def bucket(key: str) -> int:
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big") % 100


def partition(data: DatasetDict) -> dict[str, list[dict]]:
    """Samples by train/validation/test. Unknown split names (halueval's `data`) join the train
    pool; partitions the source lacks are carved from that pool by a hash of the content.
    Every model splits through here, so their test sets are the same rows."""
    parts: dict[str, list[dict]] = {name: [] for name in PARTITIONS}
    for split, rows in data.items():
        parts[SPLIT_ALIASES.get(split, "train")].extend(rows)
    carve = [name for name in ("test", "validation") if not parts[name]]
    if not carve:
        return parts
    pool, parts["train"] = parts["train"], []
    for sample in pool:
        b = bucket(content_key(sample))
        for i, name in enumerate(carve):
            if HOLDOUT_PERCENT * i <= b < HOLDOUT_PERCENT * (i + 1):
                parts[name].append(sample)
                break
        else:
            parts["train"].append(sample)
    return parts


def load_samples(names: list[str], data_dir: Path, part: str, limit: int) -> dict[str, list[dict]]:
    """{dataset: samples} in `part`. `limit` keeps that many per dataset, chosen by a hash of the content, so every
    run scores the same ones."""
    loaded = {}
    for name in names:
        samples = [s for s in partition(load_from_disk(data_dir / name))[part] if len(s["options"]) >= 2]
        if limit and len(samples) > limit:
            samples = sorted(samples, key=lambda s: hashlib.sha1(f"{content_key(s)}:{s['question']}".encode()).digest())
            samples = samples[:limit]
        if samples:
            loaded[name] = samples
    return loaded
