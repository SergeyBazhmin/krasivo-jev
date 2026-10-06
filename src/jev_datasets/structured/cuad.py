import json
import random
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, map_records, record
from jev_datasets.utils import disjoint_contexts, pick_question, stable_shuffle, text_choice

REVISION = "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2"
URL = f"https://raw.githubusercontent.com/The-Atticus-Project/cuad/{REVISION}/data.zip"
FIELDS = {
    "Document Name": "the contract's title",
    "Agreement Date": "the date on which the agreement was made or signed",
    "Effective Date": "the explicitly stated date on which the agreement takes effect",
    "Governing Law": "the explicitly named governing law or the clause stating it",
}
QUESTIONS = [
    "Which quoted text in this excerpt gives {field}?",
    "Select the literal text specifying {field} in the contract excerpt.",
    "What does this excerpt explicitly state for {field}? Use the original wording.",
    "Find {field} as written in the excerpt.",
    "Choose the wording for {field} that appears in the supplied contract passage.",
]
DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)\b"
    r"|\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b",
    re.IGNORECASE,
)


def cuad_records(path: str) -> Iterator[dict]:
    with zipfile.ZipFile(path) as archive:
        test_titles = {row["title"] for row in json.loads(archive.read("test.json"))["data"]}
        # The full file preserves all answers together; train_separate_questions splits multi-answer annotations.
        contracts = json.loads(archive.read("CUADv1.json"))["data"]
        for contract in contracts:
            paragraphs = contract["paragraphs"]
            group = content_group([" ".join(paragraph["context"].split()) for paragraph in paragraphs])
            split = (
                "test"
                if contract["title"] in test_titles
                else "validation"
                if int(group[:8], 16) % 10 == 0
                else "train"
            )
            yield {**record({"paragraphs": paragraphs}, group), "split": split}


def explicit_answers(paragraph: dict) -> Iterator[tuple[str, str, str]]:
    context = paragraph["context"]
    for qa in paragraph["qas"]:
        category = re.search(r'related to "([^"]+)"', qa["question"])
        if not category or (field := category[1]) not in FIELDS:
            continue
        for answer in qa["answers"]:
            text, start = answer["text"], answer["answer_start"]
            if not 3 <= len(text) <= 200 or start < 0 or context[start : start + len(text)] != text:
                continue
            if field.endswith("Date") and (
                len(text) > 80 or not DATE.search(text) or not re.search(r"\b\d{4}\b", text)
            ):
                continue  # Relative dates, durations and references to other clauses need legal interpretation.
            if field == "Document Name" and len(text.split()) < 2:
                continue
            passage = context[max(0, start - 300) : min(len(context), start + len(text) + 300)]
            if field == "Effective Date":
                before = " ".join(context[max(0, start - 300) : start].split())
                after = " ".join(context[start + len(text) : start + len(text) + 100].split())
                explicit = re.search(r"(?:effective|takes? effect|commenc\w*)[^.!?]{0,100}$", before, re.IGNORECASE)
                named = re.match(r'\s*[,(]?\s*(?:the\s+)?["“]?effective date\b', after, re.IGNORECASE)
                if not explicit and not named:
                    continue  # Agreement dates are not automatically effective dates.
            if field == "Governing Law" and not re.search(r"govern|construed|laws? of", passage, re.IGNORECASE):
                continue
            yield field, text, passage


def cuad_samples(row: dict, pools: dict[str, list[str]]) -> list[dict]:
    paragraphs = json.loads(row["payload"])["paragraphs"]
    samples, seen = [], set()
    for paragraph in paragraphs:
        answers = list(explicit_answers(paragraph))
        for field, answer, passage in stable_shuffle(answers, row["group"]):
            if field in seen:
                continue
            accepted = {value.strip().casefold() for key, value, _ in answers if key == field}
            seed = row["group"] + field
            candidates = random.Random(seed).sample(pools.get(field, []), min(64, len(pools.get(field, []))))
            candidates = [
                value
                for value in candidates
                if value.strip().casefold() not in accepted
                and value.casefold() not in passage.casefold()
                and answer.casefold() not in value.casefold()
                and value.casefold() not in answer.casefold()
            ]
            question = pick_question(QUESTIONS, seed).format(field=FIELDS[field])
            state = (
                "Contract excerpt. Match explicitly written text; do not infer legal consequences.\n\n[…]\n"
                + passage
                + "\n[…]"
            )
            if sample := text_choice(state, question, answer, candidates, seed):
                samples.append(sample)
                seen.add(field)
    return samples


class CUADSpansDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        return grouped_records(cuad_records, path=DownloadManager().download(URL))

    def prepare(self):
        pools = defaultdict(set)
        for row in self.data["train"]:
            for paragraph in json.loads(row["payload"])["paragraphs"]:
                for field, answer, _ in explicit_answers(paragraph):
                    pools[field].add(answer)
        pools = {field: sorted(values) for field, values in pools.items()}
        self.data = map_records(self.data, lambda row: cuad_samples(row, pools), __file__)
        self.data = disjoint_contexts(self.data)


cuad_spans_dataset = CUADSpansDataset(name="cuad_spans", hf_path="The-Atticus-Project/cuad")
