from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Do these two sentences mean the same thing?",
    "Is the second sentence a paraphrase of the first?",
    "Are these sentences paraphrases of each other?",
    "Do both sentences express the same meaning?",
    "Is the meaning of the two sentences equivalent?",
]

OPTIONS = make_options(["yes", "no"])
ABOUT = [
    (
        "You are given two sentences that share almost all their words. Decide whether they are paraphrases, that "
        "is, they say the same thing, including who did what to whom, when and where. Moving words around can "
        "change the meaning completely."
    ),
    (
        "Below are two sentences that share almost all of their words. Decide whether they are paraphrases, saying "
        "the same thing, including who did what to whom, when and where. Rearranging words can change the meaning "
        "completely."
    ),
    (
        "You are given two sentences made of nearly the same words. Decide if they are paraphrases: they must say "
        "the same thing, down to who did what to whom, when and where. Moving words around can completely change "
        "the meaning."
    ),
    (
        "You compare pairs of sentences that share almost all their words. Two sentences are paraphrases when they "
        "say the same thing, including who did what to whom, when and where; swapping words around can change the "
        "meaning completely."
    ),
    (
        "Two sentences with almost the same words are given. Decide whether they are paraphrases, meaning they say "
        "the same thing, including who did what to whom and when and where. Changing the word order can alter the "
        "meaning completely."
    ),
]


class PAWSDataset(JevDataset):
    type = DatasetType.NOUL

    def prepare(self):
        # labels: 0 = different meaning, 1 = paraphrase. Pairs are word-swapped, so high lexical
        # overlap does not give the answer away
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, text := f"Sentence 1: {x['sentence1']}\nSentence 2: {x['sentence2']}"),
                pick_question(QUESTIONS, text),
                OPTIONS,
                "yes" if x["label"] == 1 else "no",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


# `labeled_final` is human-labelled; `unlabeled_final` carries silver labels from back-translation
paws_dataset = PAWSDataset(name="paws", hf_path="google-research-datasets/paws", hf_name="labeled_final")
