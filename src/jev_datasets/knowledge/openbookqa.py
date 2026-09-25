from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class OpenBookQADataset(JevDataset):
    def prepare(self):
        # the `additional` config ships the science fact each question is built on
        self.data = self.data.map(
            lambda x: make_sample(
                x["fact1"],
                x["question_stem"],
                make_options(x["choices"]["label"], x["choices"]["text"]),
                x["answerKey"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


openbookqa_dataset = OpenBookQADataset(
    name="openbookqa", hf_path="allenai/openbookqa", hf_name="additional"
)
