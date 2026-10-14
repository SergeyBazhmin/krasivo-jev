from string import ascii_uppercase

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = [
    (
        "You are taking a logical reasoning test. Read the passage and answer the question about its argument or "
        "situation, for example what it assumes, what would strengthen or weaken it, what follows from it, or "
        "where its reasoning goes wrong."
    ),
    (
        "You are sitting a logical reasoning test. Read the passage and answer the question about its argument or "
        "situation, such as what it assumes, what would strengthen or weaken it, what follows from it, or where "
        "its reasoning goes wrong."
    ),
    (
        "Below is a passage from a logical reasoning test and a question about its argument or situation, for "
        "example what it takes for granted, what would strengthen or weaken it, what follows from it, or where its "
        "reasoning is flawed. Answer the question."
    ),
    (
        "As a test taker in a logical reasoning exam, read the passage and answer the question on its argument or "
        "situation: what it assumes, what would strengthen or weaken it, what follows from it, or where its "
        "reasoning fails."
    ),
    (
        "A logical reasoning test gives a passage and a question about its argument or situation, such as what it "
        "assumes, what would make it stronger or weaker, what follows from it, or where the reasoning goes wrong. "
        "Answer the question."
    ),
]


class ReClorDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["context"], "Passage"),
                x["question"],
                make_options(list(ascii_uppercase[: len(x["answers"])]), x["answers"]),
                ascii_uppercase[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


reclor_dataset = ReClorDataset(name="reclor", hf_path="tasksource/reclor")
