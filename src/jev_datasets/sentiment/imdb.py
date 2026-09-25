from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "What is the sentiment of this movie review?",
    "Did the reviewer like the movie?",
    "Is this movie review positive or negative?",
    "What is the reviewer's overall opinion of the film?",
    "How does the author feel about the movie?",
]

IDS = {"neg": "negative", "pos": "positive"}


class IMDBDataset(JevDataset):
    def prepare(self):
        ids = [IDS[name] for name in self.class_names()]
        options = make_options(ids)
        # the `unsupervised` split is unlabelled (label == -1)
        self.data.pop("unsupervised", None)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], pick_question(QUESTIONS, x["text"]), options, ids[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


imdb_dataset = IMDBDataset(name="imdb", hf_path="stanfordnlp/imdb")
