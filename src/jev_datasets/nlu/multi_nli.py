from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "What is the relation between the premise and the hypothesis?",
    "Does the premise entail, contradict, or say nothing about the hypothesis?",
    "Given the premise, is the hypothesis true, false, or undetermined?",
    "How does the hypothesis relate to the premise?",
    "Is the hypothesis entailed by the premise, contradicted by it, or neutral?",
]


class MultiNLIDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()
        options = make_options(names)
        # unlabelled pairs carry label == -1
        self.data = self.data.filter(lambda x: x["label"] >= 0)
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"],
                f"Hypothesis: {x['hypothesis']}\n{pick_question(QUESTIONS, x['premise'] + x['hypothesis'])}",
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


GENRE_QUESTIONS = [
    "Which genre is this text taken from?",
    "What kind of source does this passage come from?",
    "Which type of writing is this excerpt from?",
    "Where was this text most likely taken from?",
    "What is the genre of this passage?",
]

GENRES = {
    "fiction": "fiction",
    "government": "government report",
    "slate": "Slate magazine article",
    "telephone": "telephone conversation",
    "travel": "travel guide",
}


class MultiNLIGenreDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        options = make_options(list(GENRES), list(GENRES.values()))
        # mismatched genres never occur in train, so there is nothing to learn them from
        self.data.pop("validation_mismatched", None)
        # each premise is paired with ~3 hypotheses: keep it once. The hypothesis is left out,
        # since annotators wrote it and it says nothing about the source genre
        seen: set[int] = set()
        self.data = self.data.filter(lambda x: x["promptID"] not in seen and not seen.add(x["promptID"]))
        self.data = self.data.map(
            lambda x: make_sample(x["premise"], pick_question(GENRE_QUESTIONS, x["premise"]), options, x["genre"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


multi_nli_dataset = MultiNLIDataset(name="multi_nli", hf_path="nyu-mll/multi_nli")
multi_nli_genre_dataset = MultiNLIGenreDataset(name="multi_nli_genre", hf_path="nyu-mll/multi_nli")
