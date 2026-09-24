from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "What is the relation between the premise and the hypothesis?"
# order of `label_dist`, the share of 100 annotators that picked each label
OPTIONS = make_options(["entailment", "neutral", "contradiction"])


class ChaosNLIDataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"], f"Hypothesis: {x['hypothesis']}\n{QUESTION}", OPTIONS, x["label_dist"]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


chaos_nli_dataset = ChaosNLIDataset(name="chaos_nli", hf_path="tasksource/chaos-mnli-ambiguity")
