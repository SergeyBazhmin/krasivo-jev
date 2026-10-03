import re

from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

# train is a chat-style instruction/output dump; the dev sets are structured
TRAIN_PROMPT = re.compile(r"Context:\n(.*)\n\nQuestion: (.*)\n\nOptions:\n(.*)", re.DOTALL)
TRAIN_ANSWER = re.compile(r'"answer": "([A-Z])"\s*}\s*$')
OPTION = re.compile(r"([A-Z])\) (.*)")
ABOUT = (
    "You are given a logic problem: a context of facts and rules, some of them distractors, and a statement. Take the "
    "context as everything there is to know. The statement is true if it follows, false if its negation follows, and "
    "uncertain if neither does."
)


def parse_options(lines: list[str], answer: str) -> tuple[list[dict[str, str]], str]:
    """ "A) True" lines and the answer letter -> options keyed by their text, and the answer's text"""
    texts = dict(OPTION.fullmatch(line.strip()).groups() for line in lines if line.strip())
    return make_options(list(texts.values())), texts[answer]


def make_train_sample(x):
    context, question, listing = TRAIN_PROMPT.fullmatch(x["instruction"]).groups()
    options, answer = parse_options(listing.splitlines(), TRAIN_ANSWER.search(x["output"])[1])
    return make_sample(with_context(ABOUT, context.strip(), "Context"), question.strip(), options, answer)


def make_dev_sample(x):
    options, answer = parse_options(x["options"], x["answer"])
    return make_sample(with_context(ABOUT, x["context"], "Context"), x["question"], options, answer)


class ProverQADataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # the files differ in schema, so each is loaded on its own
        files = {"train": "train/provergen-5000.json"} | {
            f"dev_{level}": f"dev/{level}.json" for level in ("easy", "medium", "hard")
        }
        return DatasetDict(
            {split: load_dataset(self.hf_path, data_files=path, split="train") for split, path in files.items()}
        )

    def prepare(self):
        self.data = DatasetDict(
            {
                split: rows.map(
                    make_train_sample if split == "train" else make_dev_sample,
                    remove_columns=rows.column_names,
                    features=SAMPLE_FEATURES,
                )
                for split, rows in self.data.items()
            }
        )


proverqa_dataset = ProverQADataset(name="proverqa", hf_path="opendatalab/ProverQA")
