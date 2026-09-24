from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which ending most plausibly continues the text?"


class HellaSwagDataset(JevDataset):
    def prepare(self):
        # the `test` split is unlabelled (label == "")
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                x["ctx"],
                QUESTION,
                make_options([str(i) for i in range(len(x["endings"]))], x["endings"]),
                x["label"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


hellaswag_dataset = HellaSwagDataset(name="hellaswag", hf_path="Rowan/hellaswag")
