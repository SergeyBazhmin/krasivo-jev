import json
from collections.abc import Iterator

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, map_records, record
from jev_datasets.utils import make_option, make_sample, pick_question, stable_shuffle, with_context

REVISION = "5bbc05f234b754572779c8f09dfa706b23802d29"
URL = f"https://huggingface.co/datasets/BitAgent/bfcl_shuffle_full/resolve/{REVISION}/"
# Single-call categories, whose answer is one function, and the irrelevance ones, where none fits. Left out:
# parallel calls (several answers), multi-turn (state spread over executed calls), live_relevance (no answer key),
# and live_simple/live_multiple, which the When2Call test is built from.
CATEGORIES = {
    "simple": True,
    "multiple": True,
    "java": True,
    "javascript": True,
    "irrelevance": False,
    "live_irrelevance": False,
}
NONE = "none"
QUESTIONS = [
    "Which function should be called for this request?",
    "What function fits the user's request?",
    "Which of the functions serves the request, if any?",
    "Pick the function to call for this request.",
    "Which function call does the request need?",
]
ABOUT = [
    (
        "You route a user's request to a function. Below are the functions an AI assistant can call, then the "
        "user's request. Choose the function whose description matches what the user asks for, or none if no "
        "function can do it."
    ),
    (
        "An AI assistant can call the functions described below. Read the user's request and decide which "
        "function it needs. If none of them does what the user asks, the answer is none."
    ),
    (
        "Below are function definitions and a request from a user. Select the one function that carries out the "
        "request; when the functions only look related but cannot do what is asked, select none."
    ),
    (
        "You are the tool selector of an AI assistant. Given its functions and a user's message, pick the "
        "function that should be called, or none when no function fits the request."
    ),
    (
        "A user asks an AI assistant for something, and the assistant has the functions described below. Decide "
        "which function serves the request; a function that merely shares words with it does not count, and then "
        "the answer is none."
    ),
]


def bfcl_records(paths: dict[str, tuple[str, str | None]]) -> Iterator[dict]:
    for category, (path, answers_path) in paths.items():
        answers = {}
        if answers_path:
            with open(answers_path) as file:
                answers = {(row := json.loads(line))["id"]: row["ground_truth"] for line in file if line.strip()}
        with open(path) as file:
            for line in filter(str.strip, file):
                row = json.loads(line)
                if CATEGORIES[category]:
                    calls = answers.get(row["id"], [])
                    if len(calls) != 1:
                        continue
                    row["answer"] = next(iter(calls[0]))
                else:
                    row["answer"] = NONE
                # one question can come with a fitting function in one category and an unfitting one in another
                yield {**record(row, content_group(row["question"])), "split": "test"}


def bfcl_samples(row: dict) -> list[dict]:
    x = json.loads(row["payload"])
    names = [function["name"] for function in x["function"]]
    if len(set(names)) != len(names) or x["answer"] not in names + [NONE]:
        return []
    functions = "\n".join(json.dumps(function, ensure_ascii=False) for function in x["function"])
    dialogue = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for turn in x["question"] for m in turn)
    state = with_context(ABOUT, f"Functions:\n{functions}\n\nConversation:\n{dialogue}")
    options = [make_option(name, name) for name in stable_shuffle(names, state)]
    options.append(make_option(NONE, "none of these functions"))
    return [make_sample(state, pick_question(QUESTIONS, state), options, x["answer"])]


class BFCLDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        manager = DownloadManager()
        paths = {
            category: (
                manager.download(f"{URL}BFCL_v3_{category}.json"),
                manager.download(f"{URL}possible_answer/BFCL_v3_{category}.json") if has_answer else None,
            )
            for category, has_answer in CATEGORIES.items()
        }
        # an evaluation set only: `regroup` re-splits it 80/10/10
        return grouped_records(bfcl_records, paths=paths)

    def prepare(self):
        self.data = map_records(self.data, bfcl_samples, __file__)


bfcl_dataset = BFCLDataset(name="bfcl", hf_path="BitAgent/bfcl_shuffle_full")
