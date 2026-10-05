from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Содержит ли сообщение токсичное высказывание?",
    "Является ли этот текст токсичным?",
    "Выражает ли автор оскорбительное или агрессивное отношение?",
    "Есть ли в тексте токсичное обращение или высказывание?",
    "Можно ли считать сообщение токсичным по его содержанию?",
]
OPTIONS = make_options(["yes", "no"], ["да", "нет"])
ABOUT = (
    "Определите токсичность сообщения по его тексту. Токсичность включает оскорбления, унижение, угрозы и "
    "агрессивную брань в адрес людей или групп. Негативная оценка события или упоминание группы сами по себе "
    "не означают токсичности. Не додумывайте отсутствующую переписку и не проверяйте факты."
)


class RussianToxicityDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # Language codes are source splits, not train/test partitions. Take Russian only, not the whole mixture.
        return DatasetDict({"train": load_dataset(self.hf_path, split="ru")})

    def prepare(self):
        self.data = self.data.filter(lambda x: x["toxic"] in (0, 1) and bool(x["text"].strip()))
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Сообщение"),
                pick_question(QUESTIONS, x["text"]),
                OPTIONS,
                "yes" if x["toxic"] else "no",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


russian_toxicity_dataset = RussianToxicityDataset(
    name="russian_toxicity", hf_path="textdetox/multilingual_toxicity_dataset"
)
