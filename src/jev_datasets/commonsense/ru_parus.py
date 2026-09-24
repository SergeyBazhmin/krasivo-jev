from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTIONS = {"cause": "Что было причиной?", "effect": "Что произошло в результате?"}


class RuPARusDataset(JevDataset):
    def prepare(self):
        names = self.class_names()  # "choice1", "choice2"
        # the `test` split is unlabelled (label == -1)
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"],
                QUESTIONS[x["question"]],
                make_options(names, [x["choice1"], x["choice2"]]),
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ru_parus_dataset = RuPARusDataset(name="ru_parus", hf_path="OVHaiLLM/russian_super_glue", hf_name="parus")
