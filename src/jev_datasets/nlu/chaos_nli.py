from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.nlu.nli import NLI_ANSWERS
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is the relation between the premise and the hypothesis?",
    "Does the premise entail, contradict, or say nothing about the hypothesis?",
    "Given the premise, is the hypothesis true, false, or undetermined?",
    "How does the hypothesis relate to the premise?",
    "Is the hypothesis entailed by the premise, contradicted by it, or neutral?",
]
# order of `label_dist`, the share of 100 annotators that picked each label
OPTIONS = make_options(["entailment", "neutral", "contradiction"])
ABOUT = (
    "You are given a premise and a hypothesis. Decide whether, judging by the premise alone, the hypothesis is "
    "definitely true, definitely false, or neither. Some pairs can be read in more than one way, and then more than "
    "one answer is plausible."
)


class ChaosNLIDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["premise"], "Premise", NLI_ANSWERS),
                f"Hypothesis: {x['hypothesis']}\n{pick_question(QUESTIONS, x['premise'] + x['hypothesis'])}",
                OPTIONS,
                x["label_dist"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


chaos_nli_dataset = ChaosNLIDataset(name="chaos_nli", hf_path="tasksource/chaos-mnli-ambiguity")
