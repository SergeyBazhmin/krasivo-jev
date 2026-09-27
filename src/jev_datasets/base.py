from enum import StrEnum

from datasets import DatasetDict, load_dataset

from jev_datasets.constants import MAX_SAMPLES, ROOT_DIR
from jev_datasets.utils import stratified_limit


class DatasetType(StrEnum):
    NOUL = "noul"  # the only options are "yes" and "no"
    CHOICE = "choice"  # any other option set


class JevDataset:
    type: DatasetType

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

    def limit(self, max_samples: int = MAX_SAMPLES):
        """Caps every split at `max_samples`, stratified by label; call after `prepare`."""
        self.data = DatasetDict(
            {split: stratified_limit(rows, max_samples) for split, rows in self.data.items()}
        )

    def save(self):
        self.limit()
        self.data.save_to_disk(ROOT_DIR / self.name)
