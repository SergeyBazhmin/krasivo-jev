import re
from collections.abc import Callable

from datasets import Dataset, DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

# RuleCollection bundles several logic sets as RL prompts. Only the ones not taken from their
# own source elsewhere are used; the rest (LogicNLI, ProofWriter, FOLIO, CLUTRR, LogiQA,
# BBH Logical Deduction) would duplicate those.
ANSWER = re.compile(r"<answer>(.*)</answer>")

AR_LSAT_ABOUT = [
    (
        'You are solving an analytical reasoning puzzle (a "logic game") like those on law school admission tests. '
        "The context sets up a scheduling, ordering or grouping task under constraints; answer the question from "
        "those constraints alone."
    ),
    (
        'You are given an analytical reasoning puzzle, a "logic game" of the kind on law school admission tests. '
        "The context sets up a scheduling, ordering or grouping task with constraints; answer the question from "
        "the constraints alone."
    ),
    (
        'Below is a "logic game", an analytical reasoning puzzle like those on law school admission tests. Its '
        "context describes a scheduling, ordering or grouping task under constraints. Use only those constraints "
        "to answer the question."
    ),
    (
        'You solve analytical reasoning puzzles ("logic games") such as those on law school admission tests. The '
        "context describes a task of scheduling, ordering or grouping under constraints, and the answer follows "
        "from those constraints alone."
    ),
    (
        'This is an analytical reasoning puzzle (a "logic game") in the style of law school admission tests. The '
        "context poses a scheduling, ordering or grouping task under constraints; answer the question using only "
        "those constraints."
    ),
]
AR_LSAT_PROMPT = re.compile(r"Context: (.*)\nQuestion: (.*)\nOptions: (.*?) Please answer the question", re.DOTALL)
LETTER = re.compile(r"(?:^|\s)([A-E])\)\s")


def make_ar_lsat_sample(prompt: str, answer: str):
    context, question, listing = AR_LSAT_PROMPT.match(prompt).groups()
    # "A) one B) two C) three ..." -> ["", "A", "one", "B", "two", ...]
    parts = LETTER.split(listing)
    ids, texts = parts[1::2], [text.strip() for text in parts[2::2]]
    state = with_context(AR_LSAT_ABOUT, context, "Context")
    return make_sample(state, question, make_options(ids, texts), answer)


def unique_prompts(rows: Dataset, seen: set[str]) -> Dataset:
    """Drops prompts already in `seen` and adds the rest to it."""
    keep = []
    for i, prompt in enumerate(rows["prompt"]):
        if (text := prompt[0]["content"]) not in seen:
            seen.add(text)
            keep.append(i)
    return rows.select(keep)


class RuleCollectionDataset(JevDataset):
    type = DatasetType.CHOICE
    # its topic's benchmark pick: scored by every model, trained on only with `train_on_benchmark`
    benchmark = True

    def __init__(self, name: str, source: str, make: Callable[[str, str], dict]):
        super().__init__(name=name, hf_path="RuleReasoner/RuleCollection-32K")
        self.source = source
        self.make = make

    def load(self) -> DatasetDict:
        data = load_dataset(self.hf_path)
        # a few prompts repeat within train, and some test prompts are also in train
        seen = set()
        return DatasetDict(
            {
                split: unique_prompts(data[split].filter(lambda x: x["data_source"] == self.source), seen)
                for split in ("train", "test")
            }
        )

    def prepare(self):
        self.data = self.data.map(
            lambda x: self.make(x["prompt"][0]["content"], ANSWER.fullmatch(x["reward_model"]["ground_truth"])[1]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


# LSAT analytical reasoning ("logic games"): scheduling and assignment under stated constraints
ar_lsat_dataset = RuleCollectionDataset(name="ar_lsat", source="AR-LSAT", make=make_ar_lsat_sample)
