from string import ascii_uppercase

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class MMLUProDataset(JevDataset):
    def __init__(self, name: str, category: str):
        super().__init__(name=name, hf_path="TIGER-Lab/MMLU-Pro")
        self.category = category

    def prepare(self):
        self.data = self.data.filter(lambda x: x["category"] == self.category)
        # up to 10 options per question, keyed "A".."J"
        self.data = self.data.map(
            lambda x: make_sample(
                "",
                x["question"],
                make_options(list(ascii_uppercase[: len(x["options"])]), x["options"]),
                x["answer"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


mmlu_pro_business_dataset = MMLUProDataset(name="mmlu_pro_business", category="business")
mmlu_pro_economics_dataset = MMLUProDataset(name="mmlu_pro_economics", category="economics")
