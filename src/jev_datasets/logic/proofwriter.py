from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Given the facts and rules, is the statement true, false, or unknown?",
    "Does the theory prove the statement, disprove it, or neither?",
    "Based only on the theory above, what is the truth value of the statement?",
    "Can the statement be derived from the facts and rules, its negation, or neither?",
    "Is the statement true, false, or undetermined by the theory?",
]
# open-world assumption throughout: what the theory doesn't settle is "Unknown", not "False"
OPTIONS = make_options(["True", "False", "Unknown"])
ABOUT = [
    (
        "You are given a theory of facts and rules, and a statement. Decide whether the theory proves the "
        "statement, disproves it, or neither. The theory is all that is known, and it is not complete: a statement "
        "it does not settle is unknown, not false."
    ),
    (
        "Below are a theory of facts and rules and a statement. Decide whether the theory proves the statement, "
        "disproves it, or neither. The theory is all that is known and it is incomplete, so a statement it does "
        "not settle is unknown rather than false."
    ),
    (
        "You are given a theory made of facts and rules, and a statement. Decide whether the statement is proved "
        "by the theory, disproved by it, or neither. Nothing is known beyond the theory, which is not complete: a "
        "statement it does not settle is unknown, not false."
    ),
    (
        "You check statements against a theory of facts and rules. Decide whether the theory proves the statement, "
        "disproves it, or does neither. Treat the theory as everything known but not as complete: what it leaves "
        "unsettled is unknown, not false."
    ),
    (
        "A theory of facts and rules is followed by a statement. Decide if the theory proves it, disproves it, or "
        "neither. Only the theory is known, and it is not complete, so a statement it does not settle is unknown "
        "and not false."
    ),
]
ANSWERS = {
    "True": "the statement can be proven from the theory",
    "False": "the negation of the statement can be proven from the theory",
    "Unknown": "neither can be proven",
}


class ProofWriterDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["theory"], "Theory", ANSWERS),
                f"Statement: {x['question']}\n{pick_question(QUESTIONS, x['theory'] + x['question'])}",
                OPTIONS,
                x["answer"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


proofwriter_dataset = ProofWriterDataset(name="proofwriter", hf_path="tasksource/proofwriter")
