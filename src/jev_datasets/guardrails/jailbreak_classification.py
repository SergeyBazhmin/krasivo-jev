from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Is this user prompt a jailbreak attempt?"
OPTIONS = make_options(["yes", "no"])


class JailbreakClassificationDataset(JevDataset):
    def prepare(self):
        # `type` is "jailbreak" / "benign"
        self.data = self.data.map(
            lambda x: make_sample(
                x["prompt"], QUESTION, OPTIONS, "yes" if x["type"] == "jailbreak" else "no"
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


jailbreak_classification_dataset = JailbreakClassificationDataset(
    name="jailbreak_classification", hf_path="jackhhao/jailbreak-classification"
)
