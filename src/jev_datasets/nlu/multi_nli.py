from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "What is the relation between the premise and the hypothesis?"


class MultiNLIDataset(JevDataset):
    def prepare(self):
        names = self.class_names()
        options = make_options(names)
        # unlabelled pairs carry label == -1
        self.data = self.data.filter(lambda x: x["label"] >= 0)
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"],
                f"Hypothesis: {x['hypothesis']}\n{QUESTION}",
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


multi_nli_dataset = MultiNLIDataset(name="multi_nli", hf_path="nyu-mll/multi_nli")
