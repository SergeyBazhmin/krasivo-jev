from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

OPTIONS = make_options(["yes", "no"], ["да", "нет"])


class RuRWSDDataset(JevDataset):
    def prepare(self):
        # the `test` split is unlabelled (label == -1)
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                x["text"],
                f"Относится ли «{x['span2_text']}» к «{x['span1_text']}»?",
                OPTIONS,
                "yes" if x["label"] else "no",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ru_rwsd_dataset = RuRWSDDataset(name="ru_rwsd", hf_path="OVHaiLLM/russian_super_glue", hf_name="rwsd")
