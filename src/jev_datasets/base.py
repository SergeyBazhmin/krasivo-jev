from enum import StrEnum
from pathlib import Path

from datasets import DatasetDict, load_dataset
from loguru import logger

from jev_datasets.constants import MAX_SAMPLES, ROOT_DIR
from jev_datasets.utils import canonical_splits, disjoint_contexts, random_limit, split_missing


class DatasetType(StrEnum):
    NOUL = "noul"  # the only options are "yes" and "no" ("да" and "нет" in Russian)
    CHOICE = "choice"  # any other option set


class JevDataset:
    type: DatasetType
    # a held-out benchmark: saved with the usual splits, but training skips it (unless asked to), and a model that
    # never trained on it is scored on all its rows
    benchmark: bool = False

    def __init__(self, name: str, hf_path: str, hf_name: str | None = None):
        self.name = name
        self.hf_path = hf_path
        self.hf_name = hf_name
        self._data: DatasetDict | None = None

    @property
    def data(self) -> DatasetDict:
        if self._data is None:
            self._data = self.load()
        return self._data

    @data.setter
    def data(self, value: DatasetDict):
        self._data = value

    def load(self) -> DatasetDict:
        return load_dataset(self.hf_path, self.hf_name)

    @property
    def source_columns(self) -> list[str]:
        """Raw columns of the dataset, dropped once the samples are built."""
        return list(next(iter(self.data.values())).column_names)

    def class_names(self, column: str = "label") -> list[str]:
        return next(iter(self.data.values())).features[column].names

    def prepare(self):
        raise NotImplementedError

    def add_type(self):
        """Stamps the dataset's `type` on every sample; call after `prepare`."""
        self.data = DatasetDict(
            {split: rows.add_column("type", [self.type.value] * rows.num_rows) for split, rows in self.data.items()}
        )

    def regroup(self):
        """Renames splits to train/validation/test, moves every context group into one split, so no test context is
        trained on, drops duplicate rows and carves the splits the source lacks; call after `prepare`."""
        self.data, moved, dropped = disjoint_contexts(canonical_splits(self.data))
        if moved:
            logger.info(f"[{self.name}] moved " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in sorted(moved.items())))
        if dropped:
            logger.info(
                f"[{self.name}] dropped duplicates " + ", ".join(f"{s}: {n}" for s, n in sorted(dropped.items()))
            )
        self.data, carved = split_missing(self.data)
        if carved:
            logger.info(f"[{self.name}] carved " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in sorted(carved.items())))
        if empty := [split for split, rows in self.data.items() if not rows.num_rows]:
            logger.warning(f"[{self.name}] empty splits after carving: {', '.join(empty)}")

    def limit(self, max_samples: int = MAX_SAMPLES):
        """Caps every split at `max_samples` by a seeded random sample; call after `prepare`."""
        self.data = DatasetDict({split: random_limit(rows, max_samples) for split, rows in self.data.items()})

    def save_to(self, path: Path):
        # `save_to_disk` writes an empty split as zero shards, which `load_from_disk` then fails on
        self.data.save_to_disk(path, num_shards={split: 1 for split, rows in self.data.items() if not rows.num_rows})

    def save(self):
        self.regroup()
        self.add_type()
        self.limit()
        self.save_to(ROOT_DIR / self.name)
