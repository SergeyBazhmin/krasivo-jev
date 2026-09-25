from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "What is the sentiment of this movie review snippet?",
    "Is this excerpt from a movie review positive or negative?",
    "What opinion of the film does this phrase convey?",
    "What is the sentiment of this sentence from a film review?",
    "Does this snippet speak well or badly of the movie?",
]


class SST2Dataset(JevDataset):
    def prepare(self):
        names = self.class_names()
        options = make_options(names)
        # the `test` split is unlabelled (label == -1)
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(x["sentence"], pick_question(QUESTIONS, x["sentence"]), options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


sst2_dataset = SST2Dataset(name="sst2", hf_path="stanfordnlp/sst2")
