from datasets import DatasetDict, concatenate_datasets, get_dataset_config_names, load_dataset

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Does the story entail the statement?",
    "Given the story, must the statement be true?",
    "Does the statement follow from the story?",
    "Based only on the story, is the statement true?",
    "Can the statement be concluded from the story?",
]
OPTIONS = make_options(["entailed", "not-entailed"], ["yes", "no"])
# "Sumit went to the garden because she was bored": the link from motive to action is
# commonsense, not something the story states
SKIPPED_TASKS = {"agents-motivations"}


class BabiNLIDataset(JevDataset):
    def load(self) -> DatasetDict:
        # one config per bAbI task, pooled
        tasks = [
            load_dataset(self.hf_path, task)
            for task in get_dataset_config_names(self.hf_path)
            if task not in SKIPPED_TASKS
        ]
        return DatasetDict({split: concatenate_datasets([task[split] for task in tasks]) for split in tasks[0]})

    def prepare(self):
        names = self.class_names()  # ["not-entailed", "entailed"]
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"],
                f"Statement: {x['hypothesis']}\n{pick_question(QUESTIONS, x['premise'] + x['hypothesis'])}",
                OPTIONS,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


babi_nli_dataset = BabiNLIDataset(name="babi_nli", hf_path="tasksource/babi_nli")
