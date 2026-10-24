import hashlib
import json
from pathlib import Path

from datasets import load_from_disk
from loguru import logger

from jev_model.constants import PARTITIONS

# option texts of a `noul` sample, in English or Russian
YES = ("yes", "да")
NOUL_OPTIONS = (["no", "yes"], ["да", "нет"])


def kind(options: list[str]) -> str:
    """The `type` a dataset stamps on its rows, for options that come without one (`predict`)."""
    return "noul" if sorted(text.strip().lower() for text in options) in NOUL_OPTIONS else "choice"


def built_names(data_dir: Path) -> list[str]:
    """Datasets that `jev prepare` has saved under `data_dir`."""
    return sorted(p.parent.name for p in data_dir.glob("*/dataset_dict.json"))


def resolve_names(names: list[str], data_dir: Path) -> list[str]:
    """`names`, or every built dataset when it is empty; fails on names that were never built."""
    built = built_names(data_dir)
    if missing := [name for name in names if name not in built]:
        raise FileNotFoundError(f"not built under {data_dir}: {', '.join(missing)} (run `jev prepare`)")
    return names or built


def is_benchmark(name: str) -> bool:
    from jev_datasets import datasets

    return name in datasets and datasets[name].benchmark


def benchmark_tag(name: str) -> str:
    """Appended to a dataset's line in printed scores, so benchmarks stand out from the other datasets."""
    return "  [benchmark]" if is_benchmark(name) else ""


def benchmark_names(data_dir: Path) -> list[str]:
    """Built benchmarks that no model trains on, but every model is scored on."""
    return [name for name in built_names(data_dir) if is_benchmark(name)]


def train_names(names: list[str], data_dir: Path, benchmark: bool = False) -> list[str]:
    """`resolve_names` without the benchmark datasets, unless `benchmark`; fails when one is named without it."""
    resolved = resolve_names(names, data_dir)
    benchmarks = [name for name in resolved if is_benchmark(name)]
    if benchmark:
        if benchmarks:
            logger.info(
                f"training on {len(benchmarks)} benchmark datasets too (train_on_benchmark): {', '.join(benchmarks)}"
            )
        return resolved
    if named := [name for name in names if is_benchmark(name)]:
        raise ValueError(f"benchmark datasets are trained on only with train_on_benchmark: {', '.join(named)}")
    if benchmarks:
        logger.info(
            f"skipped {len(benchmarks)} benchmark datasets in training (set train_on_benchmark to include them): "
            + ", ".join(benchmarks)
        )
    return [name for name in resolved if name not in benchmarks]


def content_key(sample: dict) -> str:
    # what `load_samples` orders its subsample by
    return sample["state"] or sample["question"]


def partition(path: Path, part: str) -> list[dict]:
    """Samples of `part` (train/validation/test). `jev prepare` saves exactly these splits, group-disjoint, so every
    model reads the same test rows. Only `part` is loaded: `save_to_disk` keeps each split in its own folder."""
    splits = json.loads((path / "dataset_dict.json").read_text())["splits"]
    if missing := [name for name in PARTITIONS if name not in splits]:
        raise ValueError(f"dataset lacks splits {', '.join(missing)}; rebuild it with `jev prepare`")
    return load_from_disk(path / part).to_list()


def rows(name: str, data_dir: Path, part: str) -> list[dict]:
    """Samples of `part` in the dataset `name` built under `data_dir`."""
    return partition(data_dir / name, part)


def load_samples(names: list[str], data_dir: Path, part: str, limit: int) -> dict[str, list[dict]]:
    """{dataset: samples} in `part`. `limit` keeps that many per dataset, chosen by a hash of the content, so every
    run scores the same ones."""
    loaded = {}
    for name in names:
        samples = [s for s in rows(name, data_dir, part) if len(s["options"]) >= 2]
        if limit and len(samples) > limit:
            samples = sorted(samples, key=lambda s: hashlib.sha1(f"{content_key(s)}:{s['question']}".encode()).digest())
            samples = samples[:limit]
        if samples:
            loaded[name] = samples
    return loaded
