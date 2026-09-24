from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Следует ли гипотеза из текста?"
OPTIONS = make_options(["entailment", "not_entailment"], ["да", "нет"])


class RuTERRaDataset(JevDataset):
    def prepare(self):
        names = self.class_names()
        # the `test` split is unlabelled (label == -1)
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"], f"Гипотеза: {x['hypothesis']}\n{QUESTION}", OPTIONS, names[x["label"]]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ru_terra_dataset = RuTERRaDataset(name="ru_terra", hf_path="OVHaiLLM/russian_super_glue", hf_name="terra")
