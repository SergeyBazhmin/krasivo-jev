from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class MMLUDataset(JevDataset):
    def prepare(self):
        ids = self.class_names("answer")  # "A".."D"
        self.data = self.data.map(
            lambda x: make_sample(
                # e.g. "high_school_biology"; empty for the `auxiliary_train` split
                x["subject"].replace("_", " "),
                x["question"],
                make_options(ids, x["choices"]),
                ids[x["answer"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


mmlu_dataset = MMLUDataset(name="mmlu", hf_path="cais/mmlu", hf_name="all")
