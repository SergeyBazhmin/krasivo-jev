from string import ascii_uppercase

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, stable_shuffle


class TruthfulQADataset(JevDataset):
    """`mc1` has one true answer; `mc2` several, which share the probability mass evenly."""

    def __init__(self, name: str, targets: str):
        super().__init__(name=name, hf_path="truthfulqa/truthful_qa", hf_name="multiple_choice")
        self.targets = targets

    def prepare(self):
        self.data = self.data.map(self.to_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES)

    def to_sample(self, x: dict) -> dict:
        targets = x[f"{self.targets}_targets"]
        # the source always lists the true answers first
        answers = stable_shuffle(list(zip(targets["choices"], targets["labels"])), x["question"])
        truths = sum(label for _, label in answers)
        return make_sample(
            "",
            x["question"],
            make_options(list(ascii_uppercase[: len(answers)]), [text for text, _ in answers]),
            [label / truths for _, label in answers],
        )


truthful_qa_mc1_dataset = TruthfulQADataset(name="truthful_qa_mc1", targets="mc1")
truthful_qa_mc2_dataset = TruthfulQADataset(name="truthful_qa_mc2", targets="mc2")
