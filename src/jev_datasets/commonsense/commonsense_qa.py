from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class CommonsenseQADataset(JevDataset):
    def prepare(self):
        # the `test` split ships with an empty answerKey
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                x["question_concept"],
                x["question"],
                make_options(x["choices"]["label"], x["choices"]["text"]),
                x["answerKey"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


commonsense_qa_dataset = CommonsenseQADataset(name="commonsense_qa", hf_path="tau/commonsense_qa")
