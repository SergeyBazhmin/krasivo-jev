from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = [
    (
        'You are given a short story about members of one family. "How is X related to Y?" asks what X is to Y: '
        '"How is Nicholas related to Ashley?" is answered "son" when Nicholas is Ashley\'s son. The relation is '
        "never stated outright; work it out by chaining the relations the story does state (the brother of "
        "someone's daughter is their son). Answers are gendered, and names and pronouns show gender."
    ),
    (
        'Below is a short story about members of one family. "How is X related to Y?" asks what X is to Y: "How is '
        'Nicholas related to Ashley?" is answered "son" when Nicholas is Ashley\'s son. The story never states '
        "the relation outright, so work it out by chaining the relations it does state (the brother of someone's "
        "daughter is their son). Answers are gendered, and names and pronouns show gender."
    ),
    (
        'You are given a story about the members of a family. A question "How is X related to Y?" asks what X is '
        'to Y; for example, if Nicholas is Ashley\'s son, "How is Nicholas related to Ashley?" is answered '
        '"son". The relation is not stated directly: derive it by chaining the stated relations (someone\'s '
        "daughter's brother is their son). Answers are gendered; names and pronouns tell the gender."
    ),
    (
        'You work out family relations from a short story about one family. "How is X related to Y?" means what X '
        'is to Y, so "How is Nicholas related to Ashley?" gets "son" when Nicholas is Ashley\'s son. Since the '
        "story never says the relation outright, combine the relations it gives step by step (the brother of "
        "someone's daughter is that person's son). The answers are gendered, and gender shows in names and "
        "pronouns."
    ),
    (
        'A short story describes members of a family. Answer "How is X related to Y?" with what X is to Y: when '
        'Nicholas is Ashley\'s son, "How is Nicholas related to Ashley?" is answered "son". The asked relation '
        "is never stated, so chain the relations the story states (the brother of someone's daughter is their "
        "son). Answers are gendered, with gender shown by names and pronouns."
    ),
]


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
