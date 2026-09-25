from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Based on the premises, is the conclusion true, false, or uncertain?",
    "Does the conclusion follow from the premises, contradict them, or neither?",
    "Given only the premises, what is the truth value of the conclusion?",
    "Is the conclusion entailed by the premises, refuted by them, or left open?",
    "Taking the premises as true, is the conclusion true, false, or uncertain?",
]
OPTIONS = make_options(["True", "False", "Uncertain"])


class FOLIODataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                x["nl premises"],
                f"Conclusion: {x['nl conclusion']}\n{pick_question(QUESTIONS, x['nl premises'] + x['nl conclusion'])}",
                OPTIONS,
                x["label"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


# FOLIO with its mislabelled and ambiguous examples corrected; same stories and splits
folio_dataset = FOLIODataset(name="folio", hf_path="yfxiao/folio-refined")
