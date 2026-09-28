from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Based on the premises, is the conclusion true, false, or uncertain?",
    "Does the conclusion follow from the premises, contradict them, or neither?",
    "Given only the premises, what is the truth value of the conclusion?",
    "Is the conclusion entailed by the premises, refuted by them, or left open?",
    "Taking the premises as true, is the conclusion true, false, or uncertain?",
]
OPTIONS = make_options(["True", "False", "Uncertain"])
ABOUT = (
    "You are given premises in natural language and a conclusion. Take the premises as true and as everything there is "
    "to know, ignoring outside knowledge even when they are about real things, and decide whether the conclusion "
    "follows from them."
)
ANSWERS = {
    "True": "the conclusion follows from the premises",
    "False": "the negation of the conclusion follows from the premises",
    "Uncertain": "the premises settle neither",
}


class FOLIODataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["nl premises"], "Premises", ANSWERS),
                f"Conclusion: {x['nl conclusion']}\n{pick_question(QUESTIONS, x['nl premises'] + x['nl conclusion'])}",
                OPTIONS,
                x["label"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


# FOLIO with its mislabelled and ambiguous examples corrected; same stories and splits
folio_dataset = FOLIODataset(name="folio", hf_path="yfxiao/folio-refined")
