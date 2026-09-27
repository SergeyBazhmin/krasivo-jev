from string import ascii_uppercase

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class ReClorDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                x["context"],
                x["question"],
                make_options(list(ascii_uppercase[: len(x["answers"])]), x["answers"]),
                ascii_uppercase[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


reclor_dataset = ReClorDataset(name="reclor", hf_path="tasksource/reclor")
