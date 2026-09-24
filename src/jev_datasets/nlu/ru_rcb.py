from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Каково отношение гипотезы к тексту?"
OPTIONS = make_options(
    ["entailment", "contradiction", "neutral"], ["следует из текста", "противоречит тексту", "не связана с текстом"]
)


class RuRCBDataset(JevDataset):
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


ru_rcb_dataset = RuRCBDataset(name="ru_rcb", hf_path="OVHaiLLM/russian_super_glue", hf_name="rcb")
