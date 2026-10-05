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
    "Насколько совпадает смысл двух текстов?",
    "Являются ли тексты точными, приблизительными перефразировками или не перефразировками?",
    "Как соотносятся значения этих двух формулировок?",
    "Выберите степень смыслового совпадения текстов.",
    "Передают ли тексты одну и ту же информацию?",
]
OPTIONS = make_options(["1", "0", "-1"], ["точные перефразировки", "близкие перефразировки", "не перефразировки"])
ABOUT = (
    "Сравните два новостных заголовка по смыслу, не проверяя новости по внешним источникам. "
    "Точные перефразировки передают одинаковую информацию; близкие имеют общий смысл, но различаются деталями; "
    "не перефразировки передают разную информацию."
)


class RuParaPhraserDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        root = f"https://huggingface.co/datasets/{self.hf_path}/resolve/main"
        # plus.jsonl is automatically labelled and overlaps the manually annotated headline source; do not pool it.
        data = load_dataset("json", data_files={"train": f"{root}/train.jsonl", "test": f"{root}/test.jsonl"})
        # A headline can occur in several different pairs; pair-level deduplication misses this leakage.
        held_out = set(data["test"]["text_1"]) | set(data["test"]["text_2"])
        data["train"] = data["train"].filter(lambda row: not held_out.intersection((row["text_1"], row["text_2"])))
        return data

    def prepare(self):
        # '-1' is a valid non-paraphrase class in this source, not an unlabelled row.
        self.data = self.data.filter(lambda x: str(x["class"]) in {"-1", "0", "1"})
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, "\n".join(sorted((x["text_1"], x["text_2"]))), "Тексты"),
                pick_question(QUESTIONS, "\n".join(sorted((x["text_1"], x["text_2"])))),
                OPTIONS,
                str(x["class"]),
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )
        self.data = disjoint_contexts(self.data)


ru_paraphraser_dataset = RuParaPhraserDataset(name="ru_paraphraser", hf_path="merionum/ru_paraphraser")
