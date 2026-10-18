from string import ascii_uppercase

from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = [
    (
        "You are taking an English listening exam. Read the dialogue between a man (M) and a woman (W) and answer "
        "the question about it."
    ),
    (
        "Below is a dialogue from an English exam, with M for the man and W for the woman. Answer the question "
        "about what was said."
    ),
    (
        "A dialogue between a man (M) and a woman (W) is followed by a question. Answer it from what the speakers "
        "say and imply."
    ),
    (
        "As a student in an English exam, read the conversation, where M is the man and W the woman, and answer "
        "the question about it."
    ),
    (
        "An English exam gives a dialogue, with the man marked M and the woman marked W, and a question about it. "
        "Answer the question from the dialogue."
    ),
]


def make_dream_sample(x: dict) -> dict:
    ids = list(ascii_uppercase[: len(x["choice"])])
    return make_sample(
        with_context(ABOUT, "\n".join(x["dialogue"]), "Dialogue"),
        x["question"],
        make_options(ids, x["choice"]),
        ids[x["choice"].index(x["answer"])],
    )


class DREAMDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # the main branch is a loading script, which `datasets` no longer runs; the Hub's parquet export
        return load_dataset(self.hf_path, revision="refs/convert/parquet")

    def prepare(self):
        self.data = self.data.filter(lambda x: x["answer"] in x["choice"])
        self.data = self.data.map(
            make_dream_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES, load_from_cache_file=False
        )


dream_dataset = DREAMDataset(name="dream", hf_path="dataset-org/dream")
