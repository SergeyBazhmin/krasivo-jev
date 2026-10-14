import hashlib
from pathlib import Path

from datasets import DatasetDict, load_from_disk

from jev_model.constants import PARTITIONS


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
    # what `load_samples` orders its subsample by
    return sample["state"] or sample["question"]


def partition(data: DatasetDict) -> dict[str, list[dict]]:
    """Samples by train/validation/test. `jev prepare` saves exactly these splits, group-disjoint, so every model
    reads the same test rows."""
    if missing := [name for name in PARTITIONS if name not in data]:
        raise ValueError(f"dataset lacks splits {', '.join(missing)}; rebuild it with `jev prepare`")
    return {name: list(data[name]) for name in PARTITIONS}


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
