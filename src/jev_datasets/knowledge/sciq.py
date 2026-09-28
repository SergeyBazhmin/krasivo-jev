import random

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

IDS = ["A", "B", "C", "D"]
ABOUT = (
    "You are taking a science exam. Read the passage from a textbook and answer the question; the passage contains the "
    "evidence for the answer."
)


def make_sciq_sample(x):
    # the source always lists the correct answer first; shuffle so position doesn't give it
    # away, seeded per question to stay stable. Ids are positional because a few distractors
    # repeat the correct answer's text.
    order = [0, 1, 2, 3]
    random.Random(x["question"]).shuffle(order)
    answers = [x["correct_answer"], x["distractor1"], x["distractor2"], x["distractor3"]]
    return make_sample(
        with_context(ABOUT, x["support"], "Passage"),
        x["question"],
        make_options(IDS, [answers[i] for i in order]),
        IDS[order.index(0)],
    )


class SciQDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # without its support passage a question needs outside science knowledge
        self.data = self.data.filter(lambda x: x["support"].strip())
        self.data = self.data.map(
            make_sciq_sample,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


sciq_dataset = SciQDataset(name="sciq", hf_path="allenai/sciq")
