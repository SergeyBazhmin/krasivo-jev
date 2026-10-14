import json
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset, DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
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
NLI_ABOUT = [
    (
        "Даны текст и утверждение. Судите только по тексту, без внешних фактов. Если утверждение обязательно "
        "верно, оно следует из текста; если обязательно неверно, противоречит ему. Если текст не позволяет "
        "определить его истинность, ответ — «не определено»."
    ),
    (
        "Ниже текст и утверждение. Оценивайте только по тексту, без внешних фактов. Утверждение следует из текста, "
        "если оно обязательно верно, и противоречит ему, если обязательно неверно. Если по тексту нельзя "
        "определить, верно ли оно, ответ — «не определено»."
    ),
    (
        "Определите, как утверждение соотносится с текстом, не привлекая внешних фактов. Обязательно верное "
        "утверждение следует из текста, обязательно неверное — противоречит ему, а если текст не позволяет решить, "
        "ответ — «не определено»."
    ),
    (
        "Вы проверяете утверждения по тексту и не используете внешние факты. Если утверждение обязательно верно, "
        "оно следует из текста; если обязательно неверно — противоречит ему; если его истинность по тексту не "
        "установить — «не определено»."
    ),
    (
        "Даны текст и утверждение. Опираясь только на текст, решите: утверждение следует из него (обязательно "
        "верно), противоречит ему (обязательно неверно) или «не определено», когда текст не позволяет судить о его "
        "истинности."
    ),
]
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
DANETQA_ABOUT = [
    "Ответьте на вопрос только по приведённому отрывку.",
    "Ответьте на вопрос, опираясь только на приведённый отрывок.",
    "Ниже отрывок текста. Ответьте на вопрос по нему, не используя других сведений.",
    "Используйте только приведённый отрывок, чтобы ответить на вопрос.",
    "Прочитайте отрывок и ответьте на вопрос только на его основе.",
]
ENTAILMENT_ABOUT = [
    (
        "Считайте текст верным. Ответьте «да», только если утверждение обязательно следует из текста. Ответьте "
        "«нет», если оно противоречит тексту или текст не позволяет сделать такой вывод."
    ),
    (
        "Примите текст за истину. Ответ «да» — только если утверждение обязательно из него следует. Если "
        "утверждение противоречит тексту или из текста такой вывод сделать нельзя, ответ «нет»."
    ),
    (
        "Текст считается верным. Ответьте «да», если утверждение непременно следует из текста, и «нет», если оно "
        "противоречит тексту или текст не даёт оснований для такого вывода."
    ),
    (
        "Исходите из того, что текст верен. Утверждение получает «да», только когда оно обязательно следует из "
        "текста; если оно противоречит тексту или не выводится из него, ответ — «нет»."
    ),
    (
        "Считайте написанное в тексте правдой. «Да» — если утверждение обязательно вытекает из текста; «нет» — "
        "если противоречит ему или такой вывод из текста сделать нельзя."
    ),
]
MUSERC_ABOUT = [
    (
        "Проверьте предложенный ответ на вопрос только по тексту. У вопроса может быть несколько верных ответов; "
        "каждый кандидат проверяется отдельно."
    ),
    (
        "Проверьте по тексту, верен ли предложенный ответ на вопрос. Верных ответов у вопроса может быть "
        "несколько, поэтому каждый кандидат оценивается отдельно."
    ),
    (
        "Оцените предложенный ответ на вопрос, опираясь только на текст. Вопрос может иметь несколько верных "
        "ответов; каждый вариант проверяйте независимо."
    ),
    (
        "Ниже текст. Определите, правилен ли предложенный ответ на вопрос к нему, используя только текст. "
        "Кандидаты проверяются по отдельности: верных ответов может быть несколько."
    ),
    (
        "Вы проверяете ответы на вопросы по тексту. У вопроса бывает несколько верных ответов, поэтому каждый "
        "кандидат оценивается сам по себе и только по тексту."
    ),
]
RUSSE_ABOUT = [
    "Сравните значения слова в двух контекстах. Сравниваемые употребления выделены квадратными скобками.",
    (
        "Сопоставьте значения слова в двух контекстах. Употребления, которые нужно сравнить, выделены квадратными "
        "скобками."
    ),
    (
        "Ниже два предложения с одним и тем же словом, выделенным квадратными скобками. Сравните, в каком значении оно "
        "употреблено в каждом."
    ),
    (
        "Определите, совпадают ли значения слова в двух контекстах. Сравниваемые употребления отмечены квадратными "
        "скобками."
    ),
    "Сравните смысл выделенного квадратными скобками слова в двух предложениях.",
]


def danetqa_samples(row: dict) -> list[dict]:
    if row.get("label") not in (True, False) or not row["passage"].strip():
        return []
    state = with_context(DANETQA_ABOUT, row["passage"], "Текст")
    question = f"{row['question']}\n{pick_question(QA_QUESTIONS, row['passage'] + row['question'])}"
    return [make_sample(state, question, YES_NO, "yes" if row["label"] else "no")]


def entailment_samples(row: dict) -> list[dict]:
    if row.get("label") not in ("entailment", "not_entailment"):
        return []
    premise = row.get("premise", row.get("sentence1", ""))
    hypothesis = row.get("hypothesis", row.get("sentence2", ""))
    state = with_context(ENTAILMENT_ABOUT, premise, "Текст")
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
    state = with_context(MUSERC_ABOUT, passage["text"], "Текст")
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
    state = with_context(RUSSE_ABOUT, "\n".join(sentences))
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


danetqa_dataset = RussianSuperGlueDataset(name="danetqa", task="DaNetQA", convert=danetqa_samples)
lidirus_dataset = RussianSuperGlueDataset(name="lidirus", task="LiDiRus", convert=lidirus_samples)
muserc_dataset = RussianSuperGlueDataset(name="muserc", task="MuSeRC", convert=muserc_samples)
rcb_dataset = RussianSuperGlueDataset(name="rcb", task="RCB", convert=rcb_samples)
russe_dataset = RussianSuperGlueDataset(name="russe", task="RUSSE", convert=russe_samples)
terra_dataset = RussianSuperGlueDataset(name="terra", task="TERRa", convert=entailment_samples)

# PARus and RWSD need unstated commonsense; RuCoS cloze answers can need external facts. They are left out.
