import json
import random
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator

from datasets import Dataset, DatasetDict, DownloadManager

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
STATEMENT_QUESTIONS = [
    "Which statement agrees with the table?",
    "Which description of the table is correct?",
    "Which sentence is supported by the table?",
    "Choose the statement that matches the table's cells.",
    "Which of these sentences describes the table accurately?",
]


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def totto_records(path: str, annotated: bool = False) -> Iterator[dict]:
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
                    if annotated:
                        # Statements come from the annotators' final, table-supported sentences; test has none.
                        payload["section"] = row["table_section_title"]
                        payload["sentences"] = [s["final_sentence"] for s in row.get("sentence_annotations", [])]
                    yield {**record(payload, content_group(row["table"])), "split": split}


def simple_table(table: list[list[dict]]) -> tuple[list[str], list[list[str]]] | None:
    """Headers and cell values of a small rectangular table with one header row, else None."""
    if len(table) < 3 or len(table) > 40 or not 2 <= len(table[0]) <= 12:
        return None
    width = len(table[0])
    # Flattening merged cells silently changes column meanings. Retain simple rectangular tables only.
    if any(
        len(line) != width or any(cell["column_span"] != 1 or cell["row_span"] != 1 for cell in line) for line in table
    ):
        return None
    if not all(cell["is_header"] for cell in table[0]) or any(cell["is_header"] for line in table[1:] for cell in line):
        return None
    headers = [clean(cell["value"]) for cell in table[0]]
    if not all(headers) or len({header.casefold() for header in headers}) != width:
        return None
    return headers, [[clean(cell["value"]) for cell in line] for line in table[1:]]


def table_text(title: str, headers: list[str], values: list[list[str]]) -> str:
    return f"Table: {title}\n" + "\n".join([" | ".join(headers), *(" | ".join(line) for line in values)])


def totto_samples(row: dict) -> list[dict]:
    payload = json.loads(row["payload"])
    if not (parsed := simple_table(payload["table"])):
        return []
    headers, values = parsed
    width = len(headers)
    # Prefer a named row key, but a unique numeric index is also a fully specified row identifier.
    keys = [
        i
        for i in range(width)
        if all(line[i] for line in values) and len({line[i].casefold() for line in values}) == len(values)
    ]
    if not keys:
        return []
    key = next((i for i in keys if any(re.search(r"[A-Za-z]", line[i]) for line in values)), keys[0])
    state = "Read the table using its column headers. Use only the supplied cell values.\n\n"
    state += table_text(payload["title"], headers, values)
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


def statement(row: dict) -> dict | None:
    """The table as a state, plus what distractor selection needs, for one annotated example."""
    payload = json.loads(row["payload"])
    if not payload["sentences"] or not (parsed := simple_table(payload["table"])):
        return None
    headers, values = parsed
    title = " — ".join(part for part in (payload["title"], payload["section"]) if part)
    state = "Read the table using its column headers. Decide which statement the cell values support.\n\n"
    state += table_text(title, headers, values)
    if len(state) > 16000:
        return None
    return {
        "group": row["group"],
        "page": payload["title"],
        "shape": " | ".join(headers).casefold(),
        "state": state,
        "sentence": clean(payload["sentences"][0]),
    }


def statement_samples(rows: list[dict]) -> Dataset:
    # Distractors are sentences written for other tables of the same split, so held-out sentences stay held out.
    # Tables with the same headers come first: their sentences have the same shape and vocabulary, so the cell
    # values have to decide. Other tables of the same table or page are skipped, as their sentences may hold here.
    by_shape = defaultdict(list)
    for row in rows:
        by_shape[row["shape"]].append(row)
    samples = []
    for row in rows:
        seed = row["group"] + row["sentence"]
        rng = random.Random(f"statements:{seed}")
        distractors = []
        for pool in (by_shape[row["shape"]], rows):
            for other in rng.sample(pool, min(len(pool), 12)):
                if other["group"] != row["group"] and other["page"] != row["page"] and len(distractors) < 3:
                    distractors.append(other["sentence"])
        question = pick_question(STATEMENT_QUESTIONS, seed)
        if sample := text_choice(row["state"], question, row["sentence"], distractors, seed):
            samples.append(sample)
    return Dataset.from_list(samples, features=SAMPLE_FEATURES)


class ToTToDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # Test has no reference summaries, but this task computes exact labels from public cells, not summaries.
        return grouped_records(totto_records, path=DownloadManager().download(URL))

    def prepare(self):
        self.data = self.data.map(
            explode(totto_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


class ToTToStatementsDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # Test records are loaded only so that tables shared with test stay out of train, as in totto_cells.
        data = grouped_records(totto_records, path=DownloadManager().download(URL), annotated=True)
        return DatasetDict({split: data[split] for split in ("train", "validation")})

    def prepare(self):
        self.data = DatasetDict(
            {split: statement_samples(list(filter(None, map(statement, rows)))) for split, rows in self.data.items()}
        )


totto_cells_dataset = ToTToDataset(name="totto_cells", hf_path="GEM/totto")
totto_statements_dataset = ToTToStatementsDataset(name="totto_statements", hf_path="GEM/totto")
