import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, map_records, record
from jev_datasets.utils import disjoint_contexts, pick_question, stable_shuffle, text_choice

REVISION = "2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b"
BASE_URL = f"https://raw.githubusercontent.com/kris927b/SkillSpan/{REVISION}/data/json"
QUESTIONS = [
    "Which skill or area of knowledge is explicitly mentioned in this job excerpt? Use the exact wording.",
    "Choose a skill phrase that appears literally in the job excerpt.",
    "Which of these skill descriptions is present, word for word, in the excerpt?",
    "What skill or knowledge phrase does the posting explicitly contain?",
    "Select a skill mentioned in the posting, using its own words.",
]


def skill_phrases(row: dict) -> list[str]:
    """Both BIO layers supply literal spans; the task never asks for a skill-taxonomy label."""
    tokens = row["tokens"]
    phrases = set()
    for column in ("tags_skill", "tags_knowledge"):
        tags = row[column]
        if len(tags) != len(tokens) or any(tag not in {"B", "I", "O"} for tag in tags):
            return []
        start = None
        for index, tag in enumerate([*tags, "O"]):
            if tag != "I" and start is not None:
                phrases.add(" ".join(tokens[start:index]))
                start = None
            if tag == "B":
                start = index
            elif tag == "I" and start is None:
                return []  # A broken BIO chain must not silently change the annotated span.
    return sorted(phrase for phrase in phrases if 3 <= len(phrase) <= 160)


def skillspan_records(paths: dict[str, str]) -> Iterator[dict]:
    for split, path in paths.items():
        jobs = defaultdict(list)
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            # Source ids restart in every split, so content identifies the underlying job.
            jobs[row["source"], row["idx"]].append(
                {key: row[key] for key in ("tokens", "tags_skill", "tags_knowledge")}
            )
        for sentences in jobs.values():
            group = content_group([sentence["tokens"] for sentence in sentences])
            yield {**record({"sentences": sentences}, group), "split": split}


def skillspan_samples(row: dict, pool: list[str]) -> list[dict]:
    samples = []
    sentences = json.loads(row["payload"])["sentences"]
    for sentence in stable_shuffle(sentences, row["group"]):
        phrases = skill_phrases(sentence)
        if not phrases:
            continue
        text = " ".join(sentence["tokens"])
        seed = text + ":skillspan"
        answer = stable_shuffle(phrases, seed)[0]
        # Incomplete annotations are harmless here: every negative is absent from the entire excerpt.
        candidates = random.Random(seed).sample(pool, min(64, len(pool)))
        candidates = [value for value in candidates if value.casefold() not in text.casefold()]
        similar = [value for value in candidates if abs(len(value.split()) - len(answer.split())) <= 1]
        if len(similar) >= 3:
            candidates = similar  # Prefer comparable phrase lengths so short technology names are not a shortcut.
        state = "Job-posting excerpt. Match an explicitly mentioned phrase; do not infer additional skills.\n\n" + text
        if sample := text_choice(state, pick_question(QUESTIONS, seed), answer, candidates, seed):
            samples.append(sample)
    return samples


class SkillSpanDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        paths = DownloadManager().download(
            {
                split: f"{BASE_URL}/{file}.json"
                for split, file in (("train", "train"), ("validation", "dev"), ("test", "test"))
            }
        )
        return grouped_records(skillspan_records, paths=paths)

    def prepare(self):
        pool = sorted(
            {
                phrase
                for row in self.data["train"]
                for sentence in json.loads(row["payload"])["sentences"]
                for phrase in skill_phrases(sentence)
            }
        )
        self.data = map_records(self.data, lambda row: skillspan_samples(row, pool), __file__)
        self.data = disjoint_contexts(self.data)


skillspan_dataset = SkillSpanDataset(name="skillspan", hf_path="kris927b/SkillSpan")
