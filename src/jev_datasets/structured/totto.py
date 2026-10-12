import json
import re
import zipfile
from collections.abc import Iterator

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, record
from jev_datasets.utils import SAMPLE_FEATURES, explode, pick_question, stable_shuffle, text_choice

URL = "https://storage.googleapis.com/totto-public/totto_data.zip"
QUESTIONS = [
    "What is {column} for the row whose {key} is {name}?",
    "Which value appears under {column} where {key} is {name}?",
    "Read the {column} entry for {key} = {name}.",
    "Choose the {column} value in the row identified by {key} = {name}.",
    "According to the table, what is {column} when {key} is {name}?",
]


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def totto_records(path: str) -> Iterator[dict]:
    with zipfile.ZipFile(path) as archive:
        for split, name in {
            "train": "totto_data/totto_train_data.jsonl",
            "validation": "totto_data/totto_dev_data.jsonl",
            "test": "totto_data/unlabeled_totto_test_data.jsonl",
        }.items():
            with archive.open(name) as source:
                for line in source:
                    row = json.loads(line)
                    # Only the table is input. Original summaries and highlight annotations never reveal targets.
                    payload = {"table": row["table"], "title": row["table_page_title"]}
                    yield {**record(payload, content_group(row["table"])), "split": split}


def totto_samples(row: dict) -> list[dict]:
    payload = json.loads(row["payload"])
    table = payload["table"]
    if len(table) < 3 or len(table) > 40 or not 2 <= len(table[0]) <= 12:
        return []
    width = len(table[0])
    # Flattening merged cells silently changes column meanings. Retain simple rectangular tables only.
    if any(
        len(line) != width or any(cell["column_span"] != 1 or cell["row_span"] != 1 for cell in line) for line in table
    ):
        return []
    if not all(cell["is_header"] for cell in table[0]) or any(cell["is_header"] for line in table[1:] for cell in line):
        return []
    headers = [clean(cell["value"]) for cell in table[0]]
    if not all(headers) or len({header.casefold() for header in headers}) != width:
        return []
    values = [[clean(cell["value"]) for cell in line] for line in table[1:]]
    # Prefer a named row key, but a unique numeric index is also a fully specified row identifier.
    keys = [
        i
        for i in range(width)
        if all(line[i] for line in values) and len({line[i].casefold() for line in values}) == len(values)
    ]
    if not keys:
        return []
    key = next((i for i in keys if any(re.search(r"[A-Za-z]", line[i]) for line in values)), keys[0])
    lines = [" | ".join(headers), *(" | ".join(line) for line in values)]
    state = "Read the table using its column headers. Use only the supplied cell values.\n\n"
    state += f"Table: {payload['title']}\n" + "\n".join(lines)
    if len(state) > 16000:
        return []
    samples = []
    coordinates = [(i, j) for i, line in enumerate(values) for j, value in enumerate(line) if j != key and value]
    for i, j in stable_shuffle(coordinates, row["group"]):
        answer = values[i][j]
        if len(answer) > 200:
            continue
        seed = row["group"] + f":{i}:{j}"
        question = pick_question(QUESTIONS, seed).format(column=headers[j], key=headers[key], name=values[i][key])
        distractors = [line[j] for index, line in enumerate(values) if index != i and len(line[j]) <= 200]
        if sample := text_choice(state, question, answer, distractors, seed):
            samples.append(sample)
        if len(samples) == 3:
            break
    return samples


class ToTToDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # Test has no reference summaries, but this task computes exact labels from public cells, not summaries.
        return grouped_records(totto_records, path=DownloadManager().download(URL))

    def prepare(self):
        self.data = self.data.map(
            explode(totto_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


totto_cells_dataset = ToTToDataset(name="totto_cells", hf_path="GEM/totto")
