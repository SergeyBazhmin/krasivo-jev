from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which emotion does this comment express?"


class GoEmotionsDataset(JevDataset):
    def prepare(self):
        # `labels` is a list of ClassLabel: a comment can carry several emotions
        names = next(iter(self.data.values())).features["labels"].feature.names
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(
                x["text"],
                QUESTION,
                options,
                # spread the mass evenly over every emotion the raters picked
                [x["labels"].count(i) / len(x["labels"]) for i in range(len(names))],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


go_emotions_dataset = GoEmotionsDataset(
    name="go_emotions", hf_path="google-research-datasets/go_emotions", hf_name="simplified"
)
