from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Given the facts and rules, is the statement true, false, or unknown?",
    "Does the theory prove the statement, disprove it, or neither?",
    "Based only on the theory above, what is the truth value of the statement?",
    "Can the statement be derived from the facts and rules, its negation, or neither?",
    "Is the statement true, false, or undetermined by the theory?",
]
# open-world assumption throughout: what the theory doesn't settle is "Unknown", not "False"
OPTIONS = make_options(["True", "False", "Unknown"])


class ProofWriterDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                x["theory"],
                f"Statement: {x['question']}\n{pick_question(QUESTIONS, x['theory'] + x['question'])}",
                OPTIONS,
                x["answer"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


proofwriter_dataset = ProofWriterDataset(name="proofwriter", hf_path="tasksource/proofwriter")
