import random

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Which emotion does this comment express?",
    "How does the author of this comment feel?",
    "What emotion is conveyed in this Reddit comment?",
    "Which emotion best describes this comment?",
    "What is the emotional tone of this comment?",
]
ABOUT = [
    (
        "You are given a comment from Reddit, an online discussion forum, with people's names masked as [NAME]. "
        "Decide which emotion it expresses, out of 27 fine-grained emotions such as admiration, amusement, "
        "annoyance, gratitude, nervousness or remorse, or neutral if it expresses none."
    ),
    (
        "Below is a comment from the online discussion forum Reddit, with people's names masked as [NAME]. Decide "
        "which emotion it expresses, out of 27 fine-grained emotions such as admiration, amusement, annoyance, "
        "gratitude, nervousness or remorse, or neutral if it expresses none."
    ),
    (
        "You are given a Reddit comment (Reddit is an online discussion forum) in which names are replaced by "
        "[NAME]. Classify the emotion it expresses into one of 27 fine-grained emotions, for example admiration, "
        "amusement, annoyance, gratitude, nervousness or remorse, or as neutral when there is none."
    ),
    (
        "You tag comments from Reddit, an online forum, by emotion; people's names in them are masked as [NAME]. "
        "Choose among 27 fine-grained emotions, such as admiration, amusement, annoyance, gratitude, nervousness "
        "or remorse, or neutral if the comment expresses no emotion."
    ),
    (
        "A comment was posted on Reddit, an online discussion forum, and names in it are masked as [NAME]. Decide "
        "which of 27 fine-grained emotions it expresses (admiration, amusement, annoyance, gratitude, nervousness, "
        "remorse and so on), or neutral if it expresses none."
    ),
]
NOUL_QUESTIONS = [
    "Does this comment express {emotion}?",
    "Is {emotion} conveyed in this comment?",
    "Does the author of this comment show {emotion}?",
    "Can {emotion} be found in this comment?",
    "Is this comment an expression of {emotion}?",
]
NOUL_ABOUT = [
    (
        "You are given a comment from Reddit, an online discussion forum, with people's names masked as [NAME]. "
        "Decide whether it expresses the given emotion. A comment can express several emotions at once, or none."
    ),
    (
        "Below is a comment from the online discussion forum Reddit, with people's names masked as [NAME]. Decide "
        "whether the comment expresses the given emotion; it may express several emotions, or none at all."
    ),
    (
        "You are given a Reddit comment (Reddit is an online discussion forum) in which names are replaced by "
        "[NAME]. Say whether it conveys the given emotion. One comment can convey several emotions, or none."
    ),
    (
        "You tag comments from Reddit, an online forum, by emotion; people's names in them are masked as [NAME]. "
        "Decide whether the comment expresses the given emotion. Several emotions can occur together, or none."
    ),
    (
        "A comment was posted on Reddit, an online discussion forum, and names in it are masked as [NAME]. Does "
        "it express the given emotion? A comment may express more than one emotion, or none of them."
    ),
]
YES_NO = make_options(["yes", "no"])
# Asked about per comment besides its own emotions; all 27 would make ~95% of the rows "no".
NEGATIVES = 2


def go_emotions_noul_samples(row: dict, names: list[str]) -> list[dict]:
    text = row["text"]
    # `neutral` means "none of the 27", so it is not an emotion to ask about.
    emotions = [name for name in names if name != "neutral"]
    positive = {names[i] for i in row["labels"]} - {"neutral"}
    # Unlisted emotions are those fewer than two raters agreed on, so "no" means "not agreed on".
    negative = random.Random(f"noul:{text}").sample([e for e in emotions if e not in positive], NEGATIVES)
    state = with_context(NOUL_ABOUT, text, "Comment")
    return [
        make_sample(
            state,
            pick_question(NOUL_QUESTIONS, text + emotion).format(emotion=emotion),
            YES_NO,
            "yes" if emotion in positive else "no",
        )
        for emotion in sorted(positive) + negative
    ]


class GoEmotionsDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # `labels` is a list of ClassLabel: a comment can carry several emotions
        names = next(iter(self.data.values())).features["labels"].feature.names
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Comment"),
                pick_question(QUESTIONS, x["text"]),
                options,
                # one of the emotions the raters picked, chosen per comment so the target stays one-hot
                names[random.Random(f"emotion:{x['text']}").choice(x["labels"])],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


go_emotions_dataset = GoEmotionsDataset(
    name="go_emotions", hf_path="google-research-datasets/go_emotions", hf_name="simplified"
)


class GoEmotionsNoulDataset(JevDataset):
    type = DatasetType.NOUL

    def prepare(self):
        names = next(iter(self.data.values())).features["labels"].feature.names
        self.data = self.data.map(
            explode(lambda row: go_emotions_noul_samples(row, names)),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


go_emotions_noul_dataset = GoEmotionsNoulDataset(
    name="go_emotions_noul", hf_path="google-research-datasets/go_emotions", hf_name="simplified"
)
