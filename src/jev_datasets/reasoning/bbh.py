import re

from datasets import Dataset, DatasetDict, load_dataset

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

# Only tasks whose answer follows from the text alone; left out are those that lean on world
# or cultural knowledge (movie_recommendation, ruin_names, sports_understanding, snarks,
# hyperbaton, date_understanding, causal_judgement, disambiguation_qa, geometric_shapes,
# salient_translation_error_detection) and the free-form ones (dyck_languages, word_sorting,
# object_counting, multistep_arithmetic_two).
# tasks that list their options as "(A) ..." or "- ..." lines after "Options:"
LISTED_TASKS = [
    "formal_fallacies",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "navigate",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
]
# tasks with an implicit fixed answer set
IMPLICIT_OPTIONS = {
    "boolean_expressions": ["True", "False"],
    "web_of_lies": ["Yes", "No"],
}
# for tasks whose input asks no question of its own
QUESTIONS = {
    "boolean_expressions": [
        "What does this Boolean expression evaluate to?",
        "Is this Boolean expression True or False?",
        "What is the value of this expression?",
        "Evaluate the Boolean expression.",
        "What is the result of evaluating this expression?",
    ],
    "logical_deduction": [
        "Which statement about the arrangement is true?",
        "Given the paragraph, which of the options must hold?",
        "Which option is consistent with the described order?",
        "Which of these statements follows from the description?",
        "According to the paragraph, which option is correct?",
    ],
    "tracking_shuffled_objects": [
        "Which option correctly completes the last sentence?",
        "How does the final sentence end?",
        "After all the swaps, which option is right?",
        "Which option completes the story's last sentence correctly?",
        "What is the correct ending of the final sentence?",
    ],
}
OPTIONS_MARKER = "\nOptions:\n"
LETTERED = re.compile(r"\(([A-Z])\)\s*(.*)")


def task_questions(task: str) -> list[str]:
    return QUESTIONS[task.rsplit("_", 2)[0] if task.endswith("_objects") else task]


def split_problem(text: str, task: str) -> tuple[str, str]:
    """(state, question): the question the input asks itself, else one made up for the task."""
    text = text.removeprefix("Question: ").removesuffix(" is")
    head, sep, last = text.rpartition("\n")
    if sep and last.strip().endswith("?"):  # a closing question line
        return head.strip(), last.strip()
    head, sep, last = text.rpartition(". ")
    if sep and last.strip().endswith("?"):  # a closing question sentence
        return head + ".", last.strip()
    head, sep, rest = text.partition("? ")
    if sep:  # an opening question sentence, as in `navigate`
        return rest.strip(), head + "?"
    return text.strip(), pick_question(task_questions(task), text)


def parse_input(text: str, task: str) -> tuple[str, str, list[dict[str, str]]]:
    """(state, question, options) from a task input with its options inlined."""
    if task in IMPLICIT_OPTIONS:
        return *split_problem(text, task), make_options(IMPLICIT_OPTIONS[task])
    problem, _, listing = text.partition(OPTIONS_MARKER)
    ids, texts = [], []
    for line in listing.strip().splitlines():
        if lettered := LETTERED.fullmatch(line.strip()):
            ids.append(lettered[1])
            texts.append(lettered[2])
        else:
            ids.append(line.strip().removeprefix("- "))
            texts.append(ids[-1])
    return *split_problem(problem, task), make_options(ids, texts)


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
            lambda x: any(option["id"] == parse_target(x["target"]) for option in parse_input(x["input"], task)[2])
        )
        return split.map(
            lambda x: make_sample(*parse_input(x["input"], task), parse_target(x["target"])),
            remove_columns=split.column_names,
            features=SAMPLE_FEATURES,
        )


bbh_dataset = BBHDataset(name="bbh", hf_path="lukaemon/bbh")
