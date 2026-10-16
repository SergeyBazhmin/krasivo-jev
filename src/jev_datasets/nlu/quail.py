from string import ascii_uppercase

from datasets import DatasetDict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = [
    (
        "You are taking a reading comprehension test. Read the text and answer the question about it. Some "
        "questions cannot be answered from the text; for those, choose 'not enough information'."
    ),
    (
        "Below is a text, from fiction, news, a blog or a user story, with a question about it. Answer from the "
        "text; if the text does not give the answer, choose 'not enough information'."
    ),
    (
        "Read the text and answer the question using only what it says or clearly implies. When the text does not "
        "settle the question, the answer is 'not enough information'."
    ),
    (
        "As a reader in a comprehension test, answer the question about the text below. Pick 'not enough "
        "information' when the text does not let you answer."
    ),
    (
        "A reading comprehension test gives a text and a question. Answer from the text alone, and choose 'not "
        "enough information' if it does not contain the answer."
    ),
]


class QuAILDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # the held-out set is called `challenge`, which would otherwise be merged into train
        self.data = DatasetDict({"test" if split == "challenge" else split: rows for split, rows in self.data.items()})
        # how long an event lasts is guessed from world knowledge, not read from the text
        self.data = self.data.filter(lambda x: x["question_type"] != "Event_duration")
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["context"], "Text"),
                x["question"],
                make_options(list(ascii_uppercase[: len(x["answers"])]), x["answers"]),
                ascii_uppercase[x["correct_answer_id"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
            load_from_cache_file=False,
        )


quail_dataset = QuAILDataset(name="quail", hf_path="textmachinelab/quail")
