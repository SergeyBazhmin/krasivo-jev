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
ABOUT = (
    "You are given a premise and a hypothesis. Decide whether, judging by the premise alone, the hypothesis is "
    "definitely true, definitely false, or neither. The hypothesis was written to be tricky, so the answer often "
    "hinges on a detail, a number or an implicit inference."
)


class ANLIDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()
        options = make_options(names)
        # splits are adversarial rounds: train_r1, dev_r1, test_r1, ..., test_r3
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["premise"], "Premise", NLI_ANSWERS),
                f"Hypothesis: {x['hypothesis']}\n{pick_question(QUESTIONS, x['premise'] + x['hypothesis'])}",
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


anli_dataset = ANLIDataset(name="anli", hf_path="facebook/anli")
