from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class CLUTRRDataset(JevDataset):
    def prepare(self):
        # kinship terms ("aunt", "son-in-law", ...); `query` asks how `tail` relates to `head`
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(x["story"], x["query"], options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


clutrr_dataset = CLUTRRDataset(name="clutrr", hf_path="tasksource/clutrr")
