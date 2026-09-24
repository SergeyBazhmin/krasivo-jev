from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

OPTIONS = make_options(["yes", "no"])


class BoolQDataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                x["passage"],
                x["question"],
                OPTIONS,
                "yes" if x["answer"] else "no",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


boolq_dataset = BoolQDataset(name="boolq", hf_path="google/boolq")
