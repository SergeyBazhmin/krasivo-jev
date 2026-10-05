from datasets import DatasetDict, concatenate_datasets, get_dataset_config_names, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, stable_shuffle, with_context

QUESTIONS = [
    "Какое предложение грамматически корректно?",
    "Выберите грамматически правильную формулировку.",
    "В каком предложении соблюдены нормы русского языка?",
    "Какой вариант является языково корректным?",
    "Какая из двух формулировок не содержит языковой ошибки?",
]
ABOUT = (
    "Даны два предложения с небольшим различием в форме или сочетании слов. "
    "Выберите лингвистически корректный вариант. Оценивайте русский язык, не истинность сообщаемых фактов."
)


def make_rublimp_sample(row: dict) -> dict:
    texts = (row["source_sentence"], row["target_sentence"])
    seed = "\n".join(sorted(texts))
    # The original grammatical sentence is always source_sentence; shuffle its position without exposing metadata.
    options = stable_shuffle(make_options(["A", "B"], list(texts)), seed)
    return make_sample(with_context(ABOUT, seed, "Предложения"), pick_question(QUESTIONS, seed), options, "A")


class RuBLiMPDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        tasks = sorted(get_dataset_config_names(self.hf_path))
        rows = concatenate_datasets([load_dataset(self.hf_path, task, split="train") for task in tasks])
        # The Hub calls the only split 'train', but these 45k pairs are the published evaluation benchmark.
        return DatasetDict({"test": rows})

    def prepare(self):
        self.data = self.data.filter(lambda x: x["source_sentence"] != x["target_sentence"])
        self.data = self.data.map(make_rublimp_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES)


rublimp_dataset = RuBLiMPDataset(name="rublimp", hf_path="RussianNLP/RuBLiMP")
