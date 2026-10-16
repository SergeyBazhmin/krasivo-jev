from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = [
    (
        "You are taking a science exam. You are given two facts; combine them to answer the question. The answer "
        "follows from the two facts together, not from either one alone."
    ),
    (
        "Below are two science facts and an exam question. Chain the facts together to find the answer; neither "
        "fact answers the question on its own."
    ),
    (
        "As a student in a science exam, answer the question by combining the two facts given. Each fact supplies "
        "one step, and together they give the answer."
    ),
    (
        "Two facts are given, followed by a science question. The answer comes from linking the facts: one connects "
        "the question to an intermediate idea, the other connects that idea to the answer."
    ),
    (
        "A science exam gives you two facts and a question. Combine the facts to answer it; the answer follows from "
        "both of them together."
    ),
]


class QASCDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # the test split is unlabelled and has no facts
        self.data.pop("test", None)
        # the questions need science knowledge, which the two facts supply; `combinedfact` would state the answer
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, f"1. {x['fact1']}\n2. {x['fact2']}", "Facts"),
                x["question"],
                make_options(x["choices"]["label"], x["choices"]["text"]),
                x["answerKey"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
            load_from_cache_file=False,
        )


qasc_dataset = QASCDataset(name="qasc", hf_path="allenai/qasc")
