from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is the sentiment of this movie review snippet?",
    "Is this excerpt from a movie review positive or negative?",
    "What opinion of the film does this phrase convey?",
    "What is the sentiment of this sentence from a film review?",
    "Does this snippet speak well or badly of the movie?",
]
ABOUT = [
    (
        "You are given a single sentence from a movie review, lowercased and split into tokens. Decide whether the "
        "opinion of the film it conveys is positive or negative."
    ),
    (
        "Below is a single sentence from a movie review, lowercased and split into tokens. Decide whether the "
        "opinion of the film it conveys is positive or negative."
    ),
    (
        "You are given one sentence taken from a movie review; it is lowercased and tokenized. Decide if it "
        "conveys a positive or a negative opinion of the film."
    ),
    (
        "You classify sentences from movie reviews by sentiment. The sentence below is lowercased and split into "
        "tokens; decide whether its opinion of the film is positive or negative."
    ),
    (
        "A sentence from a movie review is shown below in lowercase and split into tokens. Judge whether it "
        "expresses a positive or negative opinion of the film."
    ),
]


class SST2Dataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()
        options = make_options(names)
        # the `test` split is unlabelled (label == -1)
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["sentence"], "Sentence"),
                pick_question(QUESTIONS, x["sentence"]),
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


sst2_dataset = SST2Dataset(name="sst2", hf_path="stanfordnlp/sst2")
