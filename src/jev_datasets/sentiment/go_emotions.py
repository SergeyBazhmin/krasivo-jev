import random

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

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
