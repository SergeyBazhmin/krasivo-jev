from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "What is the relation between the premise and the hypothesis?",
    "Does the premise entail, contradict, or say nothing about the hypothesis?",
    "Given the premise, is the hypothesis true, false, or undetermined?",
    "How does the hypothesis relate to the premise?",
    "Is the hypothesis entailed by the premise, contradicted by it, or neutral?",
]


class MultiNLIDataset(JevDataset):
    def prepare(self):
        names = self.class_names()
        options = make_options(names)
        # unlabelled pairs carry label == -1
        self.data = self.data.filter(lambda x: x["label"] >= 0)
        self.data = self.data.map(
            lambda x: make_sample(
                x["premise"],
                f"Hypothesis: {x['hypothesis']}\n{pick_question(QUESTIONS, x['premise'] + x['hypothesis'])}",
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


multi_nli_dataset = MultiNLIDataset(name="multi_nli", hf_path="nyu-mll/multi_nli")
