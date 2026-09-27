import random

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Which emotion does this comment express?",
    "How does the author of this comment feel?",
    "What emotion is conveyed in this Reddit comment?",
    "Which emotion best describes this comment?",
    "What is the emotional tone of this comment?",
]


class GoEmotionsDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # `labels` is a list of ClassLabel: a comment can carry several emotions
        names = next(iter(self.data.values())).features["labels"].feature.names
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(
                x["text"],
                pick_question(QUESTIONS, x["text"]),
                options,
                # one of the emotions the raters picked, chosen per comment so the target stays one-hot
                names[random.Random(f"emotion:{x['text']}").choice(x["labels"])],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


go_emotions_dataset = GoEmotionsDataset(
    name="go_emotions", hf_path="google-research-datasets/go_emotions", hf_name="simplified"
)
