from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.nlu.nli import NLI_ANSWERS
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is the relation between the premise and the hypothesis?",
    "Does the premise entail, contradict, or say nothing about the hypothesis?",
    "Given the premise, is the hypothesis true, false, or undetermined?",
    "How does the hypothesis relate to the premise?",
    "Is the hypothesis entailed by the premise, contradicted by it, or neutral?",
]
NLI_ABOUT = [
    (
        "You are given a premise, a sentence of written or spoken English, and a hypothesis. Decide whether, "
        "judging by the premise alone, the hypothesis is definitely true, definitely false, or neither."
    ),
    (
        "Below are a premise, a sentence of written or spoken English, and a hypothesis. Judging by the premise "
        "alone, decide whether the hypothesis is definitely true, definitely false, or neither."
    ),
    (
        "You are given a hypothesis and a premise, which is a sentence of English, written or spoken. Decide "
        "whether the premise alone makes the hypothesis definitely true, definitely false, or neither."
    ),
    (
        "You judge a hypothesis against a premise, a sentence of written or spoken English. Using only the "
        "premise, decide whether the hypothesis is definitely true, definitely false, or neither."
    ),
    (
        "A premise (a sentence of written or spoken English) is followed by a hypothesis. Decide, based only on "
        "the premise, whether the hypothesis is definitely true, definitely false, or neither."
    ),
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
                with_context(NLI_ABOUT, x["premise"], "Premise", NLI_ANSWERS),
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

GENRE_ABOUT = [
    (
        "You are given a passage of English text. Decide which kind of source it comes from: a work of fiction, a "
        "US government report or web page, an article from Slate (an online magazine on news, politics and "
        "culture), a transcribed telephone conversation between strangers, or a travel guide."
    ),
    (
        "Below is a passage of English text. Decide which kind of source it comes from: a work of fiction, a US "
        "government report or web page, an article from Slate (an online magazine on news, politics and culture), "
        "a transcribed telephone conversation between strangers, or a travel guide."
    ),
    (
        "You are given an English passage. Identify its source: fiction, a US government report or web page, a "
        "Slate article (Slate is an online magazine covering news, politics and culture), a transcript of a "
        "telephone conversation between strangers, or a travel guide."
    ),
    (
        "You classify English passages by where they come from: a work of fiction, a US government report or web "
        "page, the online news, politics and culture magazine Slate, a transcribed phone call between strangers, "
        "or a travel guide. Decide the source of the passage below."
    ),
    (
        "A passage of English text is given. Decide whether it comes from a work of fiction, a US government "
        "report or web page, an article in Slate (an online magazine about news, politics and culture), a "
        "transcribed telephone conversation between two strangers, or a travel guide."
    ),
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
            lambda x: make_sample(
                with_context(GENRE_ABOUT, x["premise"], "Passage"),
                pick_question(GENRE_QUESTIONS, x["premise"]),
                options,
                x["genre"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


multi_nli_dataset = MultiNLIDataset(name="multi_nli", hf_path="nyu-mll/multi_nli")
multi_nli_genre_dataset = MultiNLIGenreDataset(name="multi_nli_genre", hf_path="nyu-mll/multi_nli")
