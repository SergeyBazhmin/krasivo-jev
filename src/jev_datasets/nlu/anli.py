from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "What is the relation between the premise and the hypothesis?"


class ANLIDataset(JevDataset):
    def prepare(self):
        names = self.class_names()
        options = make_options(names)
        # splits are adversarial rounds: train_r1, dev_r1, test_r1, ..., test_r3
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"], f"Hypothesis: {x['hypothesis']}\n{QUESTION}", options, names[x["label"]]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


anli_dataset = ANLIDataset(name="anli", hf_path="facebook/anli")
