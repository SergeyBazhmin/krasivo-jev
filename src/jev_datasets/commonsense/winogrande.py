from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which option correctly fills the blank (_)?"


class WinoGrandeDataset(JevDataset):
    def prepare(self):
        # the `test` split is unlabelled (answer == "")
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                x["sentence"],
                QUESTION,
                make_options(["1", "2"], [x["option1"], x["option2"]]),
                x["answer"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


winogrande_dataset = WinoGrandeDataset(
    name="winogrande", hf_path="allenai/winogrande", hf_name="winogrande_xl"
)
