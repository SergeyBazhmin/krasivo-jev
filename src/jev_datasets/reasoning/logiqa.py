from string import ascii_uppercase

from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class LogiQADataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # the main branch is a loading script, which `datasets` no longer runs; the Hub's parquet export
        return load_dataset(self.hf_path, revision="refs/convert/parquet")

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                x["context"],
                x["query"],
                make_options(list(ascii_uppercase[: len(x["options"])]), x["options"]),
                ascii_uppercase[x["correct_option"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


logiqa_dataset = LogiQADataset(name="logiqa", hf_path="lucasmccabe/logiqa")
