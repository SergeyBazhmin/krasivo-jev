import random
import re
from collections import Counter

from datasets import Dataset, DatasetDict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample, pick_question, with_context

# Specialist -> how annotations spell it (any case ending may follow). Synonyms and near-equivalents share a key.
SPECIALISTS = {
    "аллерголог": ["аллерголог"],
    "андролог": ["андролог"],
    "гастроэнтеролог": ["гастроэнтеролог"],
    "гематолог": ["гематолог"],
    "гинеколог": ["гинеколог"],
    "дерматолог": ["дерматолог", "дерматокосметолог"],
    "диетолог": ["диетолог", "нутрициолог"],
    "кардиолог": ["кардиолог"],
    "косметолог": ["косметолог", "дерматокосметолог"],
    "маммолог": ["маммолог", "мамолог"],
    "невролог": ["невролог", "нейролог"],
    "нефролог": ["нефролог"],
    "онколог": ["онколог"],
    "ортопед": ["ортопед"],
    "отоларинголог": ["отоларинголог", "оториноларинголог", "лор"],
    "офтальмолог": ["офтальмолог"],
    "педиатр": ["педиатр"],
    "проктолог": ["проктолог", "колопроктолог"],
    "психиатр": ["психиатр"],
    "пульмонолог": ["пульмонолог"],
    "репродуктолог": ["репродуктолог"],
    "ревматолог": ["ревматолог"],
    "стоматолог": ["стоматолог"],
    "терапевт": ["терапевт", "терпевт", "врач общей практики", "семейный врач", "семейная медицина"],
    "травматолог": ["травматолог"],
    "уролог": ["уролог"],
    "хирург": ["хирург"],
    "эндокринолог": ["эндокринолог"],
}
# From a word start, so "хирург" is not found in "нейрохирург" nor "терапевт" in "психотерапевт".
PATTERNS = {
    name: re.compile("|".join(rf"(?<![а-яё]){spelling}" for spelling in spellings))
    for name, spellings in SPECIALISTS.items()
}

QUESTIONS = [
    "К какому врачу стоит обратиться?",
    "Какой специалист нужен пациенту?",
    "К какому специалисту лучше записаться?",
    "Какого врача стоит посетить с этой проблемой?",
    "Консультация какого специалиста нужна пациенту?",
]
NOUL_QUESTIONS = [
    "Стоит ли пациенту обратиться к специалисту «{doctor}»?",
    "Подходит ли пациенту врач «{doctor}»?",
    "Нужна ли пациенту консультация специалиста «{doctor}»?",
    "Стоит ли с этой проблемой записаться к врачу «{doctor}»?",
    "Правильно ли пациенту идти к специалисту «{doctor}»?",
]
ABOUT = [
    "Ниже вопрос пациента. Определите, к какому врачу-специалисту ему стоит обратиться с этой проблемой.",
    "Пациент описывает жалобу. Решите, какой специалист занимается такими проблемами.",
    "Прочитайте обращение пациента и подберите врача, к которому с ним стоит пойти.",
    "Вы направляете пациентов к врачам. По описанию проблемы определите подходящего специалиста.",
    "Ниже вопрос человека о здоровье. Определите, консультация какого специалиста ему нужна.",
]
NOUL_ABOUT = [
    "Ниже вопрос пациента. Определите, стоит ли ему обратиться с этой проблемой к указанному врачу.",
    "Пациент описывает жалобу. Решите, занимается ли такими проблемами указанный специалист.",
    "Прочитайте обращение пациента и оцените, подходит ли для него указанный врач.",
    "Вы направляете пациентов к врачам. По описанию проблемы определите, подходит ли указанный специалист.",
    "Ниже вопрос человека о здоровье. Определите, нужна ли ему консультация указанного специалиста.",
]
OPTIONS = make_options(list(SPECIALISTS))
YES_NO = make_options(["yes", "no"], ["да", "нет"])
# Unrelated specialists asked about per question in the yes/no variant.
NEGATIVES = 2


def specialists(annotation: str) -> list[str]:
    """Specialists an annotation names: "уролог или андролог" gives both, unknown ones ("инфекционист") none."""
    annotation = annotation.strip().lower()
    # A dental subspecialty ("стоматолог-хирург") is still a dentist.
    if annotation.startswith("стоматолог"):
        return ["стоматолог"]
    return [name for name, pattern in PATTERNS.items() if pattern.search(annotation)]


def merged_questions(data: DatasetDict) -> DatasetDict:
    """One row per patient question with every annotation of it. A question repeated across splits stays in the
    held-out one, and its annotations (which sometimes disagree) are pooled."""
    votes: dict[str, list[list[str]]] = {}
    home: dict[str, str] = {}
    for split in sorted(data, key=lambda name: name != "test"):
        for row in data[split]:
            names = specialists(row["to_doctor"])
            question = row["user_question"].strip()
            if names and question:
                votes.setdefault(question, []).append(names)
                home.setdefault(question, split)
    return DatasetDict(
        {
            split: Dataset.from_list([{"question": q, "votes": v} for q, v in votes.items() if home[q] == split])
            for split in data
        }
    )


def rus_med_samples(row: dict) -> list[dict]:
    # Each annotation spreads one vote over the specialists it names.
    shares = Counter()
    for names in row["votes"]:
        for name in names:
            shares[name] += 1 / len(names) / len(row["votes"])
    text = row["question"]
    # One correct option: the most voted specialist, ties broken per question. The other named specialists are
    # dropped from the options rather than kept as wrong answers.
    tied = sorted(shares)
    random.Random(f"choice:{text}").shuffle(tied)
    answer = max(tied, key=shares.__getitem__)
    options = [option for option in OPTIONS if option["id"] == answer or option["id"] not in shares]
    return [make_sample(with_context(ABOUT, text, "Вопрос пациента"), pick_question(QUESTIONS, text), options, answer)]


def rus_med_noul_samples(row: dict) -> list[dict]:
    # A specialist is right for the share of annotations naming it; "уролог или андролог" makes both right.
    right = {name: sum(name in names for names in row["votes"]) / len(row["votes"]) for name in SPECIALISTS}
    text = row["question"]
    positive = [name for name in SPECIALISTS if right[name] > 0]
    negative = random.Random(f"noul:{text}").sample([n for n in SPECIALISTS if not right[n]], NEGATIVES)
    state = with_context(NOUL_ABOUT, text, "Вопрос пациента")
    return [
        make_sample(
            state,
            pick_question(NOUL_QUESTIONS, text + name).format(doctor=name),
            YES_NO,
            [right[name], 1 - right[name]],
        )
        for name in positive + negative
    ]


class RusMedDialoguesDataset(JevDataset):
    # Choosing a specialist for a complaint is everyday knowledge any patient has, so the question alone is the
    # state. The assistant's reply is left out: it names the specialist outright.
    type = DatasetType.CHOICE
    samples = staticmethod(rus_med_samples)

    def prepare(self):
        self.data = merged_questions(self.data)
        self.data = self.data.map(
            explode(self.samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


class RusMedDialoguesNoulDataset(RusMedDialoguesDataset):
    type = DatasetType.NOUL
    samples = staticmethod(rus_med_noul_samples)


rus_med_dialogues_dataset = RusMedDialoguesDataset(name="rus_med_dialogues", hf_path="Mykes/rus_med_dialogues")
rus_med_dialogues_noul_dataset = RusMedDialoguesNoulDataset(
    name="rus_med_dialogues_noul", hf_path="Mykes/rus_med_dialogues"
)
