import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, record
from jev_datasets.utils import SAMPLE_FEATURES, explode, pick_question, stable_shuffle, text_choice

REVISION = "8f07d3e410f322d5dbf6b68b13fbf4a4aa077fe9"
URL = f"https://codeload.github.com/nerel-ds/NEREL/zip/{REVISION}"
QUESTIONS = [
    "Какой фрагмент текста указывает {relation} для «{subject}»?",
    "Что в тексте названо как {relation} для «{subject}»?",
    "Выберите {relation} для «{subject}» по приведённому тексту.",
    "Укажите фрагмент, выражающий {relation} для «{subject}».",
    "Какой вариант соответствует отношению «{relation}» для «{subject}» в тексте?",
]
# Geopolitical, medical, ideology and implicit HQ/origin relations can come from outside knowledge.
RELATIONS = {
    "AGE_IS": "возраст",
    "AGE_DIED_AT": "возраст на момент смерти",
    "DATE_OF_BIRTH": "дату рождения",
    "DATE_OF_DEATH": "дату смерти",
    "DATE_FOUNDED_IN": "дату основания",
    "DATE_OF_CREATION": "дату создания",
    "DATE_DEFUNCT_IN": "дату прекращения существования",
    "POINT_IN_TIME": "время события",
    "START_TIME": "время начала",
    "END_TIME": "время окончания",
    "TAKES_PLACE_IN": "место события",
    "WORKPLACE": "место работы",
    "SCHOOLS_ATTENDED": "учебное заведение",
    "SPOUSE": "супруга или супругу",
    "SIBLING": "брата или сестру",
    "PARENT_OF": "ребёнка",
    "ABBREVIATION": "сокращение названия",
}


def nerel_records(path: str) -> Iterator[dict]:
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".ann") or "/NEREL-v1.0/" not in name:
                continue
            source_split = name.split("/NEREL-v1.0/")[1].split("/")[0]
            if source_split not in {"train", "dev", "test"}:
                continue
            text = archive.read(name[:-4] + ".txt").decode("utf-8")
            payload = {"text": text, "annotations": archive.read(name).decode("utf-8")}
            yield {
                **record(payload, content_group(text)),
                "split": "validation" if source_split == "dev" else source_split,
            }


def parse_annotations(text: str, annotations: str) -> tuple[dict, list]:
    entities = {}
    relations = []
    for line in annotations.splitlines():
        fields = line.split("\t")
        if line.startswith("T") and len(fields) >= 3:
            parts = fields[1].split()
            # Discontinuous or damaged offsets cannot serve as literal answer evidence.
            if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                continue
            start, end = int(parts[1]), int(parts[2])
            if text[start:end] == fields[2]:
                entities[fields[0]] = {"type": parts[0], "start": start, "end": end, "text": fields[2]}
        elif line.startswith("R") and len(fields) >= 2:
            parts = fields[1].split()
            if len(parts) == 3 and parts[0] in RELATIONS.keys() | {"ALTERNATIVE_NAME"}:
                relations.append((parts[0], parts[1].removeprefix("Arg1:"), parts[2].removeprefix("Arg2:")))
    return entities, relations


def nerel_samples(row: dict) -> list[dict]:
    payload = json.loads(row["payload"])
    text = payload["text"]
    entities, relations = parse_annotations(text, payload["annotations"])
    aliases = {id: {entity["text"].casefold()} for id, entity in entities.items()}
    # Alias annotations are used only to reject false negatives, never as external input evidence.
    changed = True
    while changed:
        changed = False
        for relation, first, second in relations:
            if relation in {"ALTERNATIVE_NAME", "ABBREVIATION"} and first in aliases and second in aliases:
                joined = aliases[first] | aliases[second]
                if joined != aliases[first] or joined != aliases[second]:
                    aliases[first] = aliases[second] = joined
                    changed = True
    positives = defaultdict(set)
    for relation, subject_id, object_id in relations:
        if subject_id in entities and object_id in entities:
            positives[relation, entities[subject_id]["text"].casefold()].add(entities[object_id]["text"].casefold())
    samples = []
    seen = set()
    for relation, subject_id, object_id in relations:
        if relation not in RELATIONS or subject_id not in entities or object_id not in entities:
            continue
        subject, answer = entities[subject_id], entities[object_id]
        key = relation, subject["text"].casefold()
        # Nested mentions often describe implicit relations. Keep separate mentions in one sentence only.
        start = min(subject["start"], answer["start"])
        end = max(subject["end"], answer["end"])
        overlap = subject["start"] < answer["end"] and answer["start"] < subject["end"]
        if key in seen or overlap or re.search(r"[.!?]\s|\n", text[start:end]):
            continue
        if len(positives[key]) != 1:
            continue
        if (
            subject["type"] in {"EVENT", "PROFESSION"}
            and len(re.findall(r"(?<!\w)" + re.escape(subject["text"]) + r"(?!\w)", text, flags=re.IGNORECASE)) != 1
        ):
            continue
        seen.add(key)
        accepted = positives[key] | aliases[object_id]
        distractors = [
            entity["text"]
            for entity in entities.values()
            if entity["type"] == answer["type"]
            # Nested mentions may name the same organization with or without a modifier.
            and not any(entity["text"].casefold() in value or value in entity["text"].casefold() for value in accepted)
            and not (entity["start"] < answer["end"] and answer["start"] < entity["end"])
            and entity["text"].casefold() != subject["text"].casefold()
        ]
        seed = row["group"] + relation + subject_id
        question = pick_question(QUESTIONS, seed).format(relation=RELATIONS[relation], subject=subject["text"])
        state = "Используйте только явно выраженные сведения текста. Ответ — дословный фрагмент текста.\n\n" + text
        if sample := text_choice(state, question, answer["text"], distractors, seed):
            samples.append(sample)
    return stable_shuffle(samples, row["group"])[:4]


class NERELDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        return grouped_records(nerel_records, path=DownloadManager().download(URL))

    def prepare(self):
        self.data = self.data.map(
            explode(nerel_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


nerel_relations_dataset = NERELDataset(name="nerel_relations", hf_path="nerel-ds/NEREL")
