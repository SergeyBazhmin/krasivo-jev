from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = (
    'You are given a short story about members of one family. "How is X related to Y?" asks what X is to Y: "How is '
    'Nicholas related to Ashley?" is answered "son" when Nicholas is Ashley\'s son. The relation is never stated '
    "outright; work it out by chaining the relations the story does state (the brother of someone's daughter is their "
    "son). Answers are gendered, and names and pronouns show gender."
)


class CLUTRRDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # kinship terms ("aunt", "son-in-law", ...); `query` asks how `tail` relates to `head`
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(with_context(ABOUT, x["story"], "Story"), x["query"], options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


clutrr_dataset = CLUTRRDataset(name="clutrr", hf_path="tasksource/clutrr")
