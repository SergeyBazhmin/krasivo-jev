import re
from itertools import pairwise

from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    disjoint_contexts,
    explode,
    make_options,
    make_sample,
    pick_question,
    stable_shuffle,
    with_context,
)

QUESTIONS = [
    "Какой фрагмент содержит ответ на вопрос?",
    "Выберите фрагмент текста с ответом на этот вопрос.",
    "В каком из фрагментов находится ответ?",
    "Какой из приведённых отрывков отвечает на вопрос?",
    "Укажите фрагмент, содержащий ответ на вопрос.",
]
ABOUT = (
    "Прочитайте текст и выберите фрагмент, содержащий ответ на вопрос. "
    "Используйте только сведения из текста. Варианты ответа — дословные фрагменты этого же текста."
)


def evidence_samples(context: str, question: str, spans: list[tuple[int, str]]) -> list[dict]:
    """Recast extractive QA as evidence selection without guessing labels for generated answer distractors."""
    spans = [
        (start, answer) for start, answer in spans if answer.strip() and context[start : start + len(answer)] == answer
    ]
    if not spans or not question.strip():
        return []
    boundaries = [
        0,
        *(match.end() for match in re.finditer(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9])|\n+", context)),
        len(context),
    ]
    fragments = [(start, end, context[start:end].strip()) for start, end in pairwise(boundaries)]
    # Gold offsets matter: the same city/name can occur in an unrelated sentence earlier in the passage.
    supported = [
        text
        for start, end, text in fragments
        if any(start <= offset and offset + len(answer) <= end for offset, answer in spans)
    ]
    if not supported:
        return []
    # Exclude every fragment containing an accepted answer, not just the first gold span.
    # Sentences elsewhere in the same passage are distractors; no other document's answers are imported.
    answers = [answer.strip().casefold() for _, answer in spans]
    distractors = list(
        dict.fromkeys(
            text for _, _, text in fragments if text and not any(answer in text.casefold() for answer in answers)
        )
    )
    seed = context + question
    distractors = stable_shuffle(distractors, seed)[:3]
    if not distractors:
        return []
    texts = [supported[0], *distractors]
    options = stable_shuffle(make_options(list("ABCD")[: len(texts)], texts), seed)
    state = with_context(ABOUT, context, "Текст")
    prompt = f"Вопрос: {question}\n{pick_question(QUESTIONS, seed)}"
    return [make_sample(state, prompt, options, "A")]


def sberquad_samples(row: dict) -> list[dict]:
    spans = list(zip(row["answers"]["answer_start"], row["answers"]["text"]))
    return evidence_samples(row["context"], row["question"], spans)


def multiq_samples(row: dict) -> list[dict]:
    inputs = row["inputs"]

    # Both documents are needed for the bridge; bridge_answers are annotations, not additional input evidence.
    def normalize(text: str) -> str:
        return re.sub(r"\\+n", "\n", text)

    prefix = normalize(inputs["support_text"]) + "\n\n"
    context = prefix + normalize(inputs["text"])
    spans = []
    for answer in row["outputs"]:
        start, text = answer["offset"], answer["segment"]
        if inputs["text"][start : start + len(text)] == text:
            spans.append((len(prefix) + len(normalize(inputs["text"][:start])), normalize(text)))
    return evidence_samples(context, inputs["question"], spans)


class SberQuADDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        data = super().load()
        # The public test split has no gold spans.
        data.pop("test", None)
        return data

    def prepare(self):
        self.data = self.data.map(
            explode(sberquad_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )
        self.data = disjoint_contexts(self.data)


class MultiQDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # MERA has a labelled train and a public test with blank answer spans.
        url = f"https://huggingface.co/datasets/{self.hf_path}/resolve/main/data/multiq/train.jsonl"
        return load_dataset("json", data_files={"train": url})

    def prepare(self):
        self.data = self.data.map(
            explode(multiq_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


mera_multiq_dataset = MultiQDataset(name="mera_multiq", hf_path="ai-forever/MERA", hf_name="multiq")
sberquad_dataset = SberQuADDataset(name="sberquad", hf_path="kuznetsoffandrey/sberquad")
