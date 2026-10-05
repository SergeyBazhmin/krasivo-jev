import json
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset, DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    disjoint_contexts,
    explode,
    make_options,
    make_sample,
    pick_question,
    with_context,
)

YES_NO = make_options(["yes", "no"], ["да", "нет"])
QA_QUESTIONS = [
    "Ответьте на вопрос, используя только текст.",
    "Каков ответ на вопрос согласно приведённому тексту?",
    "Ответьте на вопрос на основании отрывка.",
    "Дайте ответ на вопрос, опираясь на текст.",
    "Что следует из текста в ответ на этот вопрос?",
]
ENTAILMENT_QUESTIONS = [
    "Следует ли утверждение из текста?",
    "Можно ли сделать этот вывод из текста?",
    "Подтверждает ли текст данное утверждение?",
    "Обязательно ли верно утверждение, если текст верен?",
    "Является ли утверждение следствием приведённого текста?",
]
NLI_QUESTIONS = [
    "Как утверждение соотносится с текстом?",
    "Текст подтверждает утверждение, опровергает его или не даёт ответа?",
    "Следует ли утверждение из текста, противоречит ему или остаётся неопределённым?",
    "Что можно заключить об утверждении на основании текста?",
    "Утверждение верно, неверно или не определено согласно тексту?",
]
NLI_OPTIONS = make_options(
    ["entailment", "contradiction", "neutral"], ["следует из текста", "противоречит тексту", "не определено"]
)
NLI_ABOUT = (
    "Даны текст и утверждение. Судите только по тексту, без внешних фактов. "
    "Если утверждение обязательно верно, оно следует из текста; если обязательно неверно, противоречит ему. "
    "Если текст не позволяет определить его истинность, ответ — «не определено»."
)
ANSWER_QUESTIONS = [
    "Подтверждает ли текст этот ответ на вопрос?",
    "Является ли предложенный ответ верным согласно тексту?",
    "Можно ли ответить на вопрос таким образом, опираясь на текст?",
    "Соответствует ли предложенный ответ приведённому отрывку?",
    "Правилен ли данный ответ на вопрос по содержанию текста?",
]
SENSE_QUESTIONS = [
    "Одинаковое ли значение у выделенных слов в двух предложениях?",
    "Употреблено ли выделенное слово в одном и том же значении?",
    "Совпадают ли значения отмеченных слов?",
    "Означают ли выделенные слова одно и то же в этих контекстах?",
    "Совпадает ли смысл выделенного слова в обоих предложениях?",
]


def danetqa_samples(row: dict) -> list[dict]:
    if row.get("label") not in (True, False) or not row["passage"].strip():
        return []
    state = with_context("Ответьте на вопрос только по приведённому отрывку.", row["passage"], "Текст")
    question = f"{row['question']}\n{pick_question(QA_QUESTIONS, row['passage'] + row['question'])}"
    return [make_sample(state, question, YES_NO, "yes" if row["label"] else "no")]


def entailment_samples(row: dict) -> list[dict]:
    if row.get("label") not in ("entailment", "not_entailment"):
        return []
    premise = row.get("premise", row.get("sentence1", ""))
    hypothesis = row.get("hypothesis", row.get("sentence2", ""))
    state = with_context(
        "Считайте текст верным. Ответьте «да», только если утверждение обязательно следует из текста. "
        "Ответьте «нет», если оно противоречит тексту или текст не позволяет сделать такой вывод.",
        premise,
        "Текст",
    )
    question = f"Утверждение: {hypothesis}\n{pick_question(ENTAILMENT_QUESTIONS, premise + hypothesis)}"
    return [make_sample(state, question, YES_NO, "yes" if row["label"] == "entailment" else "no")]


def rcb_samples(row: dict) -> list[dict]:
    if row.get("label") not in ("entailment", "contradiction", "neutral"):
        return []
    state = with_context(NLI_ABOUT, row["premise"], "Текст")
    question = f"Утверждение: {row['hypothesis']}\n{pick_question(NLI_QUESTIONS, row['premise'] + row['hypothesis'])}"
    return [make_sample(state, question, NLI_OPTIONS, row["label"])]


def muserc_samples(row: dict) -> list[dict]:
    passage = row["passage"]
    state = with_context(
        "Проверьте предложенный ответ на вопрос только по тексту. У вопроса может быть несколько верных ответов; "
        "каждый кандидат проверяется отдельно.",
        passage["text"],
        "Текст",
    )
    samples = []
    # Multiple correct candidates are independent judgements, not probabilities over mutually exclusive answers.
    for question in passage["questions"]:
        for answer in question["answers"]:
            if answer.get("label") not in (0, 1):
                continue
            seed = passage["text"] + question["question"] + answer["text"]
            prompt = (
                f"Вопрос: {question['question']}\nПредложенный ответ: {answer['text']}\n"
                f"{pick_question(ANSWER_QUESTIONS, seed)}"
            )
            samples.append(make_sample(state, prompt, YES_NO, "yes" if answer["label"] else "no"))
    return samples


def russe_samples(row: dict) -> list[dict]:
    if row.get("label") not in (True, False):
        return []
    sentences = []
    for i in (1, 2):
        text, start, end = row[f"sentence{i}"], row[f"start{i}"], row[f"end{i}"]
        if not 0 <= start < end <= len(text):
            return []
        sentences.append(f"{text[:start]}[{text[start:end]}]{text[end:]}")
    sentences.sort()
    state = with_context(
        "Сравните значения слова в двух контекстах. Сравниваемые употребления выделены квадратными скобками.",
        "\n".join(sentences),
    )
    return [
        make_sample(
            state,
            pick_question(SENSE_QUESTIONS, "\n".join(sentences)),
            YES_NO,
            "yes" if row["label"] else "no",
        )
    ]


def lidirus_samples(row: dict) -> list[dict]:
    # Diagnostic tags explicitly identify world-knowledge and commonsense examples.
    if row.get("knowledge", "").strip():
        return []
    return entailment_samples(row)


class RussianSuperGlueDataset(JevDataset):
    # Russian да/нет options use choice; noul requires literal yes/no for the hosted baseline.
    type = DatasetType.CHOICE

    def __init__(self, name: str, task: str, convert: Callable[[dict], list[dict]]):
        super().__init__(name=name, hf_path="RussianNLP/russian_super_glue", hf_name=task)
        self.convert = convert

    def load(self) -> DatasetDict:
        # The Hub repo is a legacy loading script; read its original archives without executing it.
        url = f"https://huggingface.co/datasets/{self.hf_path}/resolve/main/data/{self.hf_name}.zip"
        root = Path(DownloadManager().download_and_extract(url)) / self.hf_name
        files = (
            {"test": "LiDiRus.jsonl"}
            if self.hf_name == "LiDiRus"
            else {"train": "train.jsonl", "validation": "val.jsonl"}
        )
        # Other tasks' public test files have no labels. LiDiRus is a labelled diagnostic test only.
        data = {}
        for split, filename in files.items():
            rows = [json.loads(line) for line in (root / filename).read_text(encoding="utf-8-sig").splitlines()]
            if self.hf_name == "LiDiRus":
                for row in rows:
                    row.setdefault("knowledge", "")
            data[split] = Dataset.from_list(rows)
        return DatasetDict(data)

    def prepare(self):
        self.data = self.data.map(
            explode(self.convert), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )
        self.data = disjoint_contexts(self.data)


danetqa_dataset = RussianSuperGlueDataset(name="danetqa", task="DaNetQA", convert=danetqa_samples)
lidirus_dataset = RussianSuperGlueDataset(name="lidirus", task="LiDiRus", convert=lidirus_samples)
muserc_dataset = RussianSuperGlueDataset(name="muserc", task="MuSeRC", convert=muserc_samples)
rcb_dataset = RussianSuperGlueDataset(name="rcb", task="RCB", convert=rcb_samples)
russe_dataset = RussianSuperGlueDataset(name="russe", task="RUSSE", convert=russe_samples)
terra_dataset = RussianSuperGlueDataset(name="terra", task="TERRa", convert=entailment_samples)

# PARus and RWSD need unstated commonsense; RuCoS cloze answers can need external facts. They are left out.
