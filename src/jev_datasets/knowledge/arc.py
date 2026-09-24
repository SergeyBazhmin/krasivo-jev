from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class ARCDataset(JevDataset):
    def prepare(self):
        # answer keys are mostly "A".."E", with a few questions keyed "1".."4"
        self.data = self.data.map(
            lambda x: make_sample(
                "",
                x["question"],
                make_options(x["choices"]["label"], x["choices"]["text"]),
                x["answerKey"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


arc_challenge_dataset = ARCDataset(
    name="arc_challenge", hf_path="allenai/ai2_arc", hf_name="ARC-Challenge"
)
arc_easy_dataset = ARCDataset(name="arc_easy", hf_path="allenai/ai2_arc", hf_name="ARC-Easy")
