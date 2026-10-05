from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    disjoint_contexts,
    make_options,
    make_sample,
    pick_question,
    with_context,
)

QUESTIONS = [
    "Является ли предложение допустимым в русском языке?",
    "Можно ли считать это предложение языково корректным?",
    "Приемлемо ли данное предложение для носителя русского языка?",
    "Построено ли предложение корректно с точки зрения русского языка?",
    "Является ли эта фраза лингвистически приемлемой?",
]
OPTIONS = make_options(["acceptable", "unacceptable"], ["приемлемое", "неприемлемое"])
ABOUT = (
    "Оцените языковую приемлемость предложения: грамматику и осмысленность формулировки. "
    "Не проверяйте истинность фактов. Необычное, но грамматически и семантически допустимое предложение приемлемо."
)


class RuCoLADataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        root = f"https://huggingface.co/datasets/{self.hf_path}/resolve/main/data"
        # dev.csv already merges the two dev subsets; loading all CSVs would duplicate validation rows.
        # Public test.csv has no acceptability labels.
        # dev.csv also has a pandas index and subset column; load the files separately before selecting input fields.
        return DatasetDict(
            {
                split: load_dataset("csv", data_files=f"{root}/{filename}", split="train").select_columns(
                    ["sentence", "acceptable", "error_type"]
                )
                for split, filename in {"train": "in_domain_train.csv", "validation": "dev.csv"}.items()
            }
        )

    def prepare(self):
        # Hallucination judgements can depend on the missing text that a generator was translating/paraphrasing.
        self.data = self.data.filter(
            lambda x: x["acceptable"] in (0, 1) and bool(x["sentence"].strip()) and x["error_type"] != "Hallucination"
        )
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["sentence"], "Предложение"),
                pick_question(QUESTIONS, x["sentence"]),
                OPTIONS,
                "acceptable" if x["acceptable"] else "unacceptable",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )
        self.data = disjoint_contexts(self.data)


rucola_dataset = RuCoLADataset(name="rucola", hf_path="RussianNLP/rucola")
