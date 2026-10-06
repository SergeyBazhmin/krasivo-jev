import json
import re
from collections import defaultdict
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, record
from jev_datasets.utils import SAMPLE_FEATURES, explode, pick_question, stable_shuffle, text_choice

ROOT = "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg"
QUESTIONS = [
    "Which description retains the recorded {relation} value for {subject}?",
    "Choose the version with the stated {relation} value for {subject}.",
    "Which version gives the supplied value of {relation} for {subject}?",
    "Which description uses the correct {relation} value for {subject}?",
    "Select the version matching the recorded {relation} of {subject}.",
]
QUESTIONS_RU = [
    "В каком описании сохранено значение поля «{relation}» для «{subject}»?",
    "Выберите вариант с указанным значением поля «{relation}» для «{subject}».",
    "Какой вариант передаёт приведённое значение поля «{relation}» для «{subject}»?",
    "В каком описании верно указано поле «{relation}» для «{subject}»?",
    "Какой вариант соответствует записи «{subject} — {relation}»?",
]


def readable(value: str) -> str:
    return value.strip().strip('"').replace("_", " ")


def webnlg_records(files: dict[str, str]) -> Iterator[dict]:
    for split, path in files.items():
        for row in json.loads(Path(path).read_text())["values"]:
            # All lexicalizations of a fact set share a group, including reordered or duplicate triples.
            facts = sorted(set(row["input"]))
            yield {**record({"facts": facts, "references": row["target"]}, content_group(facts)), "split": split}


def mentions(text: str, value: str) -> list[re.Match]:
    return list(re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, flags=re.IGNORECASE))


def webnlg_samples(row: dict, pools: dict[str, list[str]], russian: bool) -> list[dict]:
    payload = json.loads(row["payload"])
    facts = [tuple(readable(value) for value in fact.split(" | ")) for fact in payload["facts"]]
    if not facts or any(len(fact) != 3 for fact in facts):
        return []
    # Russian references translate names without providing an alignment (Aarhus -> Орхус).
    # Controlled Russian renderings preserve the supplied entity strings and require no such outside knowledge.
    references = (
        [
            f"Для «{subject}» поле «{re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', relation).lower()}» имеет значение «{value}»."
            for subject, relation, value in facts
        ]
        if russian
        else payload["references"]
    )
    samples = []
    for reference in stable_shuffle(references, row["group"]):
        for subject, relation, answer in stable_shuffle(facts, reference):
            relation_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", relation).lower()
            if russian and f"поле «{relation_text}»" not in reference:
                continue
            subject_mentions = mentions(reference, subject)
            # Only mutate an explicit object occurrence outside the subject name (e.g. Aarhus Airport / Aarhus).
            object_mentions = [
                match
                for match in mentions(reference, answer)
                if not any(other.start() < match.end() and match.start() < other.end() for other in subject_mentions)
            ]
            if not subject_mentions or len(object_mentions) != 1 or subject.casefold() == answer.casefold():
                continue
            accepted = {obj.casefold() for sub, rel, obj in facts if sub == subject and rel == relation}
            # Match the relation's value type, rather than letting broken grammar identify the gold sentence.
            candidates = [obj for _, rel, obj in facts if rel == relation] + pools.get(relation, [])
            numeric = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", answer))
            if numeric:
                candidates.extend(str(Decimal(answer) + offset) for offset in (-2, -1, 1, 2))
            candidates = [
                value
                for value in candidates
                if value.casefold() not in accepted
                and value.casefold() != subject.casefold()
                and not mentions(reference, value)
                and bool(re.fullmatch(r"-?\d+(?:\.\d+)?", value)) == numeric
            ]
            # Each negative changes one traceable value; no unrestricted LLM-generated summaries are used.
            match = object_mentions[0]
            seed = row["group"] + reference + relation
            distractors = [
                reference[: match.start()] + value + reference[match.end() :]
                for value in stable_shuffle(candidates, seed)[:8]
            ]
            lines = [f"{sub} — {re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', rel).lower()} — {obj}" for sub, rel, obj in facts]
            about = (
                "Ниже приведены факты в формате «объект — отношение — значение». "
                "Сравните только значение поля из вопроса. Остальные части описаний не оцениваются. "
                "Используйте только приведённые факты."
                if russian
                else "Facts are written as 'subject — relation — value'. "
                "Compare only the value of the relation asked about; other parts of the descriptions are not assessed. "
                "Use only the supplied information."
            )
            state = about + "\n\n" + "\n".join(lines)
            question = pick_question(QUESTIONS_RU if russian else QUESTIONS, seed).format(
                subject=subject, relation=re.sub(r"(?<=[a-z])(?=[A-Z])", " ", relation).lower()
            )
            if sample := text_choice(state, question, reference, distractors, seed):
                samples.append(sample)
            if len(samples) == 2:
                return samples
    return samples


class WebNLGDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        files = {
            split: f"{ROOT}/webnlg_{self.hf_name}_{suffix}.json"
            for split, suffix in (("train", "train"), ("validation", "val"), ("test", "test"))
        }
        return grouped_records(webnlg_records, files=DownloadManager().download(files))

    def prepare(self):
        pools = defaultdict(set)
        for row in self.data["train"]:
            for fact in json.loads(row["payload"])["facts"]:
                parts = fact.split(" | ")
                if len(parts) == 3 and 1 <= len(readable(parts[2])) <= 80:
                    pools[readable(parts[1])].add(readable(parts[2]))
        # A fixed training-only bank bounds conversion cost, with no labels borrowed from held-out sources.
        pools = {
            relation: stable_shuffle(sorted(values), f"webnlg:{relation}")[:128] for relation, values in pools.items()
        }
        self.data = self.data.map(
            explode(lambda row: webnlg_samples(row, pools, self.hf_name == "ru")),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


webnlg_en_dataset = WebNLGDataset(name="webnlg_en", hf_path="GEM/web_nlg", hf_name="en")
webnlg_ru_dataset = WebNLGDataset(name="webnlg_ru", hf_path="GEM/web_nlg", hf_name="ru")
