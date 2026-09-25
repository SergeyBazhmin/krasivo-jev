from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Is this comment toxic?",
    "Is this comment rude, disrespectful or unreasonable?",
    "Would readers find this comment toxic?",
    "Should this comment be flagged as toxic?",
    "Does this comment contain toxic language?",
]
OPTIONS = make_options(["yes", "no"])


class CivilCommentsDataset(JevDataset):
    def prepare(self):
        # `toxicity` is the fraction of raters who found the comment toxic: a soft target as is
        self.data = self.data.map(
            lambda x: make_sample(
                x["text"],
                pick_question(QUESTIONS, x["text"]),
                OPTIONS,
                [x["toxicity"], 1 - x["toxicity"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


civil_comments_dataset = CivilCommentsDataset(
    name="civil_comments", hf_path="google/civil_comments"
)
