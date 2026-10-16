import re

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    explode,
    make_options,
    make_sample,
    pick_question,
    stable_shuffle,
    with_context,
)

ABOUT = [
    (
        "You are given several Wikipedia paragraphs, most of them irrelevant, and a question comparing two "
        "entities. Find the paragraphs about both entities and answer from what they say."
    ),
    (
        "Below are Wikipedia paragraphs, only some of which matter, followed by a question that compares two "
        "entities. Locate the paragraphs about each entity and answer from them."
    ),
    (
        "Several Wikipedia paragraphs are given, mostly distractors. The question compares two entities; answer it "
        "using the paragraphs that describe them."
    ),
    (
        "You answer questions from Wikipedia paragraphs. Most of the paragraphs below are unrelated; the question "
        "compares two entities, so find what the paragraphs say about each and compare."
    ),
    (
        "Read the Wikipedia paragraphs, of which only a few are relevant, and answer the question comparing two "
        "entities, using what the paragraphs say about them."
    ),
]
NOUL_QUESTIONS = [
    "Is the answer to this question yes?",
    "Should this question be answered yes?",
    "According to the paragraphs, is the answer yes?",
    "Do the paragraphs make the answer to this question yes?",
    "Is this true according to the paragraphs?",
]
YES_NO = make_options(["yes", "no"])


def paragraphs(x: dict) -> str:
    return "\n\n".join(
        f"{title}:\n{''.join(sentences).strip()}"
        for title, sentences in zip(x["context"]["title"], x["context"]["sentences"])
    )


def entity(title: str) -> str:
    """A Wikipedia title without its disambiguation, e.g. "Ed Wood (film)" -> "Ed Wood"."""
    return re.sub(r"\s*\([^)]*\)$", "", title).strip()


def comparison_samples(x: dict) -> list[dict]:
    titles = sorted(set(x["supporting_facts"]["title"]))
    if len(titles) != 2:
        return []
    answer = x["answer"].strip()
    match = [title for title in titles if answer.casefold() in (title.casefold(), entity(title).casefold())]
    if len(match) != 1:
        return []
    texts = stable_shuffle([answer, entity(next(title for title in titles if title != match[0]))], x["id"])
    if texts[0].casefold() == texts[1].casefold():
        return []
    ids = ["A", "B"]
    return [
        make_sample(
            with_context(ABOUT, paragraphs(x), "Paragraphs"),
            x["question"],
            make_options(ids, texts),
            ids[texts.index(answer)],
        )
    ]


def yes_no_samples(x: dict) -> list[dict]:
    answer = x["answer"].strip().casefold()
    if answer not in ("yes", "no"):
        return []
    question = f"Question: {x['question']}\n{pick_question(NOUL_QUESTIONS, x['id'])}"
    return [make_sample(with_context(ABOUT, paragraphs(x), "Paragraphs"), question, YES_NO, answer)]


class HotpotQADataset(JevDataset):
    """Comparison questions whose answer is one of the two compared entities; the other is the distractor.
    Bridge questions are free-form spans with no natural distractors, so they are left out."""

    type = DatasetType.CHOICE
    samples = staticmethod(comparison_samples)

    def prepare(self):
        self.data = self.data.filter(lambda x: x["type"] == "comparison")
        self.data = self.data.map(
            explode(self.samples),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
            load_from_cache_file=False,
        )


class HotpotQANoulDataset(HotpotQADataset):
    """Comparison questions answered yes or no ("Are X and Y both ...?")."""

    type = DatasetType.NOUL
    samples = staticmethod(yes_no_samples)


hotpot_qa_dataset = HotpotQADataset(name="hotpot_qa", hf_path="hotpotqa/hotpot_qa", hf_name="distractor")
hotpot_qa_noul_dataset = HotpotQANoulDataset(name="hotpot_qa_noul", hf_path="hotpotqa/hotpot_qa", hf_name="distractor")
