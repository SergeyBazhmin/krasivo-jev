import re

from datasets import Dataset, DatasetDict, load_dataset

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

# tasks that list their options as "(A) ..." or "- ..." lines after "Options:"
LISTED_TASKS = [
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "navigate",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
]
# tasks with an implicit fixed answer set; the free-form ones (dyck_languages, word_sorting,
# object_counting, multistep_arithmetic_two) are left out
IMPLICIT_OPTIONS = {
    "boolean_expressions": ["True", "False"],
    "sports_understanding": ["yes", "no"],
    "web_of_lies": ["Yes", "No"],
}
OPTIONS_MARKER = "\nOptions:\n"
LETTERED = re.compile(r"\(([A-Z])\)\s*(.*)")


def parse_input(text: str, task: str) -> tuple[str, list[dict[str, str]]]:
    """(question, options) from a task input with its options inlined."""
    if task in IMPLICIT_OPTIONS:
        return text, make_options(IMPLICIT_OPTIONS[task])
    question, _, listing = text.partition(OPTIONS_MARKER)
    ids, texts = [], []
    for line in listing.strip().splitlines():
        if lettered := LETTERED.fullmatch(line.strip()):
            ids.append(lettered[1])
            texts.append(lettered[2])
        else:
            ids.append(line.strip().removeprefix("- "))
            texts.append(ids[-1])
    return question.strip(), make_options(ids, texts)


def parse_target(target: str) -> str:
    """"(B)" -> "B"; other targets are the option itself"""
    return target.removeprefix("(").removesuffix(")")


class BBHDataset(JevDataset):
    def load(self) -> DatasetDict:
        # one config per task, each a lone `test` split; splits here are tasks
        return DatasetDict(
            {task: load_dataset(self.hf_path, task, split="test") for task in [*LISTED_TASKS, *IMPLICIT_OPTIONS]}
        )

    def prepare(self):
        self.data = DatasetDict({task: self.prepare_task(task, split) for task, split in self.data.items()})

    @staticmethod
    def prepare_task(task: str, split: Dataset) -> Dataset:
        # 3 rows had an option split on its comma ("(A) Monsters (B) Inc") and a text target
        split = split.filter(
            lambda x: any(option["id"] == parse_target(x["target"]) for option in parse_input(x["input"], task)[1])
        )
        return split.map(
            lambda x: make_sample("", *parse_input(x["input"], task), parse_target(x["target"])),
            remove_columns=split.column_names,
            features=SAMPLE_FEATURES,
        )


bbh_dataset = BBHDataset(name="bbh", hf_path="lukaemon/bbh")
