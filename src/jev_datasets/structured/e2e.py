import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, map_records, record
from jev_datasets.utils import disjoint_contexts, pick_question, stable_shuffle, text_choice

REVISION = "164acdf986146d76f18893b3de9488725deda4c2"
BASE_URL = f"https://raw.githubusercontent.com/tuetschek/e2e-cleaning/{REVISION}/cleaned-data"
FIELDS = {
    "name": "the restaurant's name",
    "near": "the named nearby landmark",
    "food": "the cuisine",
    "eatType": "the type of establishment",
    "area": "the area",
}
QUESTIONS = [
    "What does the description explicitly say for {field}?",
    "Choose {field} as stated in the restaurant description.",
    "Which value for {field} is mentioned in the text?",
    "Identify {field} using only this description.",
    "According to the text, what is {field}?",
]
MR_FIELD = re.compile(r"([^,\[\]]+)\[([^\[\]]+)\]")


def parse_mr(text: str) -> dict[str, str] | None:
    matches = list(MR_FIELD.finditer(text))
    if not matches or MR_FIELD.sub("", text).replace(",", "").strip():
        return None
    pairs = [(match[1].strip(), match[2].strip()) for match in matches]
    if len({key for key, _ in pairs}) != len(pairs):
        return None
    return dict(pairs)


def literal(value: str, text: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, flags=re.IGNORECASE))


def e2e_records(paths: dict[str, str]) -> Iterator[dict]:
    for split, path in paths.items():
        with Path(path).open(newline="") as file:
            for row in csv.DictReader(file):
                if fields := parse_mr(row["mr"]):
                    yield {**record({"fields": fields, "text": row["ref"]}, content_group(fields)), "split": split}


def e2e_samples(row: dict, pools: dict[str, list[str]]) -> list[dict]:
    payload = json.loads(row["payload"])
    text, fields = payload["text"], payload["fields"]
    samples = []
    # Literal fields only; inferred ratings, yes/no family suitability and price normalization are omitted.
    for field in stable_shuffle(sorted(fields.keys() & FIELDS.keys()), text)[:3]:
        answer = fields[field]
        evidence = text
        if field not in {"name", "near"}:
            for name in (fields.get("name", ""), fields.get("near", "")):
                if name:
                    evidence = re.sub(re.escape(name), "", evidence, flags=re.IGNORECASE)
        if not literal(answer, evidence):
            continue
        candidates = [value for value in pools.get(field, []) if not literal(value, text)]
        seed = text + field
        question = pick_question(QUESTIONS, seed).format(field=FIELDS[field])
        state = "Restaurant description. Use only explicitly stated values.\n\n" + text
        if sample := text_choice(state, question, answer, candidates, seed):
            samples.append(sample)
    return samples


class E2ESlotsDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        paths = DownloadManager().download(
            {
                "train": f"{BASE_URL}/train-fixed.no-ol.csv",
                "validation": f"{BASE_URL}/devel-fixed.no-ol.csv",
                "test": f"{BASE_URL}/test-fixed.csv",
            }
        )
        return grouped_records(e2e_records, paths=paths)

    def prepare(self):
        pools = defaultdict(set)
        for row in self.data["train"]:
            payload = json.loads(row["payload"])
            for field, value in payload["fields"].items():
                if field in FIELDS and literal(value, payload["text"]):
                    pools[field].add(value)
        pools = {key: sorted(values) for key, values in pools.items()}
        self.data = map_records(self.data, lambda row: e2e_samples(row, pools), __file__)
        self.data = disjoint_contexts(self.data)


e2e_slots_dataset = E2ESlotsDataset(name="e2e_slots", hf_path="tuetschek/e2e_nlg_cleaned")
