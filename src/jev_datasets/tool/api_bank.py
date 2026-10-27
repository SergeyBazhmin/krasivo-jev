import hashlib
import json
import re
from collections import defaultdict

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import GROUPED_FEATURES, pick_question, text_choice, with_context

# Level 1/2 API calls; level 3 plans several searches and calls and has its own format. The response files hold
# the assistant's text replies, which can't be told apart from a list of candidates without a reference.
FILES = {
    "train": ["training-data/lv1-api-train.json", "training-data/lv2-api-train.json"],
    "test": ["test-data/level-1-api.json", "test-data/level-2-api.json"],
}
QUESTIONS = [
    "Which API request should the assistant send next?",
    "What API call comes next in this dialogue?",
    "Pick the API request that serves the user's last request.",
    "Which request should be made to the API now?",
    "Choose the next API call, with the right arguments.",
]
ABOUT = [
    (
        "You are an AI assistant that acts through APIs. Below are the APIs you can call and the dialogue so far, "
        "where earlier calls are shown as `API-Request: [...]->result`. Choose the request to send next: the API "
        "that serves the user's latest need, with argument values taken from the dialogue."
    ),
    (
        "An assistant serves the user by calling the APIs described below. Given the dialogue, where past calls "
        "and their results appear as `API-Request: [...]->result`, decide which request it must send now, with "
        "the API name and argument values the dialogue calls for."
    ),
    (
        "Below are API descriptions and a conversation between a user and an AI assistant; the assistant's past "
        "API calls appear as `API-Request: [...]->result`. Select the next API request: the right API, with its "
        "arguments filled from what the user said."
    ),
    (
        "You pick the next API call of a tool-using assistant. You get the descriptions of its APIs and the "
        "dialogue, including earlier calls written as `API-Request: [...]->result`. The right request uses the API "
        "that fits the user's current request and the argument values given in the dialogue."
    ),
    (
        "A dialogue between a user and an AI assistant follows the APIs the assistant can call. Calls it already "
        "made show up as `API-Request: [...]->result`. Decide which request comes next, meaning which API and "
        "which argument values."
    ),
]
REQUEST = re.compile(r"API-Request:\s*\[(.*)\]\s*$", re.DOTALL)
CALL_NAME = re.compile(r"(\w+)\(")
TIME = re.compile(r"The current (?:time|year) is [^\n]*")


def parse(x: dict) -> dict | None:
    text = x["instruction"] + "\n" + x["input"]
    gold = REQUEST.match((x.get("output") or x.get("expected_output") or "").strip())
    if "API descriptions:" not in text or not gold or not (name := CALL_NAME.match(gold[1])):
        return None
    apis, names, dialogue = [], [], []
    for line in text.split("API descriptions:", 1)[1].strip().splitlines():
        try:
            api = json.loads(line) if line.startswith("{") else None
        except json.JSONDecodeError:
            api = None
        if isinstance(api, dict) and (api_name := api.get("apiCode") or api.get("name")):
            apis.append(line)
            names.append(api_name)
        elif line.strip() and not line.startswith("Generate API Request"):
            dialogue.append(line)
    if not apis or not dialogue:
        return None
    time = TIME.search(text)
    return {
        "time": time[0] if time else "",
        "apis": apis,
        "names": names,
        "dialogue": "\n".join(dialogue),
        "request": gold[1].strip(),
        "name": name[1],
        "group": hashlib.sha1("\n".join(apis).encode()).hexdigest(),
    }


def make_samples(rows: Dataset) -> Dataset:
    parsed = [p for x in rows if (p := parse(x))]

    # The dialogue's other requests (often already sent, so shown in its history) make the hard distractors. Its
    # turns share the API set and first line; other dialogues on the same APIs may ask for the same thing in other
    # words (level 2 is all ToolSearcher queries), so they are left out.
    def dialogue(p: dict) -> tuple[str, str]:
        return p["group"], p["dialogue"].splitlines()[0]

    pools = defaultdict(set)
    for p in parsed:
        pools[dialogue(p)].add(p["request"])
    samples = []
    for p in parsed:
        # the same arguments sent to another API test the choice of API
        swaps = [other + p["request"][len(p["name"]) :] for other in p["names"] if other != p["name"]]
        distractors = sorted(pools[dialogue(p)] - {p["request"]}) + swaps
        state = with_context(
            ABOUT, f"{p['time']}\n\nAPIs:\n" + "\n".join(p["apis"]) + f"\n\nDialogue:\n{p['dialogue']}"
        )
        sample = text_choice(state, pick_question(QUESTIONS, state), p["request"], distractors, state)
        if sample:
            samples.append({**sample, "group": p["group"]})
    return Dataset.from_list(samples, features=GROUPED_FEATURES)


class APIBankDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # the default config mixes files of different schemas and tasks
        return DatasetDict(
            {
                split: concatenate_datasets(
                    [load_dataset(self.hf_path, data_files=path, split="train") for path in paths]
                )
                for split, paths in FILES.items()
            }
        )

    def prepare(self):
        self.data = DatasetDict({split: make_samples(rows) for split, rows in self.data.items()})


api_bank_dataset = APIBankDataset(name="api_bank", hf_path="liminghao1630/API-Bank")
