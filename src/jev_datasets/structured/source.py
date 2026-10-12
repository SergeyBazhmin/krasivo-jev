"""Raw records keep source groups intact before any one-to-many conversion."""

import hashlib
import inspect
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from datasets import Dataset, DatasetDict

from jev_datasets.utils import SAMPLE_FEATURES, explode


def conversion_version(source_file: str | None) -> str:
    """Imported helpers are cached by reference; include their code so filter edits invalidate Arrow caches."""
    files = [Path(__file__), Path(__file__).parent.parent / "utils.py"]
    if source_file and Path(source_file).is_file():
        files.append(Path(source_file))
    return hashlib.sha256(b"\n".join(file.read_bytes() for file in files)).hexdigest()


def map_records(data: DatasetDict, convert: Callable[[dict], list[dict]], source_file: str) -> DatasetDict:
    batched = explode(convert)

    def versioned(batch: dict, cache_version: str) -> dict:
        return batched(batch)

    return data.map(
        versioned,
        batched=True,
        fn_kwargs={"cache_version": conversion_version(source_file)},
        remove_columns=next(iter(data.values())).column_names,
        features=SAMPLE_FEATURES,
    )


def record(payload: dict, group: str) -> dict[str, str]:
    return {"group": group, "payload": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def content_group(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def grouped_records(generate: Callable[..., Iterable[dict]], **kwargs) -> DatasetDict:
    """Holdouts win duplicate source groups; each group has exactly one split even after conversion."""

    def versioned_generate(cache_version: str, **arguments) -> Iterable[dict]:
        yield from generate(**arguments)

    rows = Dataset.from_generator(
        versioned_generate,
        gen_kwargs={**kwargs, "cache_version": conversion_version(inspect.getsourcefile(generate))},
    )
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
