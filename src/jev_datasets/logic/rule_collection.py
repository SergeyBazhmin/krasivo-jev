import re
from collections.abc import Callable

from datasets import Dataset, DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

# RuleCollection bundles several logic sets as RL prompts. Only the ones not taken from their
# own source elsewhere are used; the rest (LogicNLI, ProofWriter, FOLIO, CLUTRR, LogiQA,
# BBH Logical Deduction) would duplicate those.
ANSWER = re.compile(r"<answer>(.*)</answer>")

PRONTOQA_PROMPT = re.compile(
    r"Rules: (.*)\nFacts: (.*)\nQuestion: (.*?)\? Please answer the question based on the given rule and facts",
    re.S,
)
PRONTOQA_QUESTIONS = [
    "Given the rules and facts, is the statement true or false?",
    "Does the statement follow from the rules and facts?",
    "Based only on the rules and facts, is the statement true?",
    "Can the statement be proven true from the rules and facts, or is it false?",
    "Taking the rules and facts as given, is the statement true or false?",
]
# the prompt offers "Unknown" too, but no ProntoQA answer is ever Unknown
PRONTOQA_OPTIONS = make_options(["True", "False"])

AR_LSAT_PROMPT = re.compile(r"Context: (.*)\nQuestion: (.*)\nOptions: (.*?) Please answer the question", re.S)
LETTER = re.compile(r"(?:^|\s)([A-E])\)\s")


def make_prontoqa_sample(prompt: str, answer: str):
    rules, facts, statement = PRONTOQA_PROMPT.match(prompt).groups()
    return make_sample(
        f"Rules: {rules}\nFacts: {facts}.",
        f"Statement: {statement}.\n{pick_question(PRONTOQA_QUESTIONS, prompt)}",
        PRONTOQA_OPTIONS,
        answer,
    )


def make_ar_lsat_sample(prompt: str, answer: str):
    context, question, listing = AR_LSAT_PROMPT.match(prompt).groups()
    # "A) one B) two C) three ..." -> ["", "A", "one", "B", "two", ...]
    parts = LETTER.split(listing)
    ids, texts = parts[1::2], [text.strip() for text in parts[2::2]]
    return make_sample(context, question, make_options(ids, texts), answer)


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


# the compositional ProntoQA variant, with made-up concept names ("wumpus", "brimpus")
prontoqa_dataset = RuleCollectionDataset(name="prontoqa", source="ProntoQA", make=make_prontoqa_sample)
# LSAT analytical reasoning ("logic games"): scheduling and assignment under stated constraints
ar_lsat_dataset = RuleCollectionDataset(name="ar_lsat", source="AR-LSAT", make=make_ar_lsat_sample)
