from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is the sentiment of this movie review?",
    "Did the reviewer like the movie?",
    "Is this movie review positive or negative?",
    "What is the reviewer's overall opinion of the film?",
    "How does the author feel about the movie?",
]

IDS = {"neg": "negative", "pos": "positive"}

ABOUT = (
    "You are given a user review of a movie. Decide whether the reviewer's overall opinion of the film is positive or "
    "negative."
)


class IMDBDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        ids = [IDS[name] for name in self.class_names()]
        options = make_options(ids)
        # the `unsupervised` split is unlabelled (label == -1)
        self.data.pop("unsupervised", None)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Review"), pick_question(QUESTIONS, x["text"]), options, ids[x["label"]]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


imdb_dataset = IMDBDataset(name="imdb", hf_path="stanfordnlp/imdb")
