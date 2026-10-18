from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = [
    (
        "You are given a fact about how one quantity affects another, and a question about a situation. Apply the "
        "fact to the situation and pick the answer; a question that ends mid-sentence asks which option completes "
        "it."
    ),
    (
        "Below is a fact describing how one property changes with another, followed by a question about a "
        "situation. Use the fact to answer; when the question stops mid-sentence, choose the option that "
        "completes it."
    ),
    (
        "A fact states how one quantity changes when another does. Apply it to the situation in the question and "
        "choose the answer. If the question is an unfinished sentence, pick the option that finishes it."
    ),
    (
        "You reason with qualitative relations. Given a fact about how one thing affects another, answer the "
        "question about a situation by applying the fact; an unfinished question asks which option completes it."
    ),
    (
        "Read the fact about how one property depends on another, then answer the question about a situation by "
        "applying that fact. A question that ends mid-sentence asks for the option that completes it."
    ),
]


class QuaRTzDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["para"], "Fact"),
                x["question"],
                make_options(x["choices"]["label"], x["choices"]["text"]),
                x["answerKey"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
            load_from_cache_file=False,
        )


quartz_dataset = QuaRTzDataset(name="quartz", hf_path="allenai/quartz")
