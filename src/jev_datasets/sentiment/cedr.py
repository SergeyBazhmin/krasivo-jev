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

EMOTIONS = ["радость", "грусть", "удивление", "страх", "гнев"]
QUESTIONS = [
    "Выражена ли в тексте эмоция «{emotion}»?",
    "Присутствует ли в тексте эмоция «{emotion}»?",
    "Передаёт ли текст эмоцию «{emotion}»?",
    "Можно ли обнаружить в тексте эмоцию «{emotion}»?",
    "Содержит ли текст выражение эмоции «{emotion}»?",
]
OPTIONS = make_options(["yes", "no"], ["да", "нет"])
ABOUT = (
    "Определите, выражена ли указанная эмоция в тексте. Радость — удовольствие или восторг; грусть — печаль "
    "или огорчение; удивление — реакция на неожиданное; страх — испуг или тревога; гнев — злость или раздражение. "
    "В тексте может быть несколько эмоций одновременно или ни одной из этих пяти. "
    "Судите по тексту, не додумывая ситуацию за его пределами."
)


def cedr_samples(row: dict) -> list[dict]:
    if not row["text"].strip() or not set(row["labels"]) <= set(range(len(EMOTIONS))):
        return []
    state = with_context(ABOUT, row["text"], "Текст")
    # Each emotion is an independent binary label; normalizing several true labels would change the task.
    return [
        make_sample(
            state,
            pick_question(QUESTIONS, row["text"] + emotion).format(emotion=emotion),
            OPTIONS,
            "yes" if i in row["labels"] else "no",
        )
        for i, emotion in enumerate(EMOTIONS)
    ]


class CEDRDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            explode(cedr_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )
        self.data = disjoint_contexts(self.data)


cedr_dataset = CEDRDataset(name="cedr", hf_path="sagteam/cedr_v1", hf_name="main")
