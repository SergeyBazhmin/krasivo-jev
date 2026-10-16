from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Does the evidence support or refute the claim?",
    "Judging by the evidence, is the claim supported, refuted, or neither?",
    "What does the evidence say about the claim?",
    "Is the claim backed by the evidence, contradicted by it, or not settled?",
    "How does the evidence bear on the claim?",
]
ABOUT = [
    (
        "You are a fact checker. You are given a sentence of evidence from a Wikipedia article and a claim. Using "
        "only the evidence, decide whether it supports the claim, refutes it, or does not give enough information. "
        "Claims often hinge on a single number, date or name."
    ),
    (
        "You check claims against evidence. Below is a sentence from a Wikipedia article, and a claim follows. "
        "Judging by the evidence alone, decide whether it supports the claim, refutes it, or leaves it open. A "
        "single number, date or name often decides the answer."
    ),
    (
        "Below is a sentence of evidence taken from a Wikipedia article, followed by a claim. Decide from the "
        "evidence alone whether the claim is supported, refuted, or not settled; watch the numbers, dates and "
        "names, which often decide it."
    ),
    (
        "As a fact checker, you compare a claim with a sentence of evidence from a Wikipedia article. Using only "
        "that evidence, say whether it supports the claim, refutes it, or lacks the information to decide. The "
        "answer often turns on a number, a date or a name."
    ),
    (
        "A claim is checked against one sentence of evidence from a Wikipedia article. Decide, based only on the "
        "evidence, whether it supports the claim, refutes it, or does not give enough information; a single "
        "number, date or name often decides it."
    ),
]
ANSWERS = {
    "supports": "the evidence shows the claim is true",
    "refutes": "the evidence shows the claim is false",
    "not enough info": "the evidence does not settle whether the claim is true or false",
}
OPTIONS = make_options(["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"], list(ANSWERS))


def make_vitaminc_sample(x: dict) -> dict:
    state = with_context(ABOUT, x["evidence"], "Evidence", ANSWERS)
    question = f"Claim: {x['claim']}\n{pick_question(QUESTIONS, x['evidence'] + x['claim'])}"
    return make_sample(state, question, OPTIONS, x["label"])


class VitaminCDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # synthetic rows are automatic edits of FEVER claims; keep the claims written against real Wikipedia revisions
        self.data = self.data.filter(lambda x: x["revision_type"] == "real")
        self.data = self.data.map(
            make_vitaminc_sample,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
            load_from_cache_file=False,
        )


vitaminc_dataset = VitaminCDataset(name="vitaminc", hf_path="tals/vitaminc")
