from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Is the claim supported by the document?",
    "Does the document back up the claim?",
    "Is everything in the claim grounded in the document?",
    "Can the claim be verified from the document?",
    "Does the document support this claim?",
]
ABOUT = [
    (
        "You check whether text written by an AI model is grounded in its source. Below is a document, and a "
        "claim follows. The claim is supported only if everything it states follows from the document; a claim "
        "that adds or changes information is not, even if it might be true."
    ),
    (
        "Below is a source document, and after it a claim from an AI-generated text. Decide whether the "
        "document supports every part of the claim. Information that is missing from the document or "
        "contradicts it makes the claim unsupported."
    ),
    (
        "You are a fact checker for AI outputs. Using only the document below, judge whether a claim is "
        "supported. It is not supported if any part of it is absent from the document or at odds with it."
    ),
    (
        "A claim taken from an AI model's output is checked against a document. Answer yes only when the "
        "document backs everything the claim says; anything unverifiable from the document means no."
    ),
    (
        "Given a document and a claim, decide whether the claim is grounded: all of its content must follow from "
        "the document. Do not use outside knowledge; claims the document doesn't support count as unsupported."
    ),
]
OPTIONS = make_options(["yes", "no"])


def make_aggrefact_sample(x: dict) -> dict:
    state = with_context(ABOUT, x["doc"].strip(), "Document")
    question = f"Claim: {x['claim'].strip()}\n{pick_question(QUESTIONS, state + x['claim'])}"
    return make_sample(state, question, OPTIONS, "yes" if x["label"] == 1 else "no")


class LLMAggreFactDataset(JevDataset):
    type = DatasetType.NOUL

    def prepare(self):
        # dev and test only: `regroup` re-splits them 80/10/10, keeping each document in one split
        self.data = self.data.map(make_aggrefact_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES)


# gated on the Hub: accept its terms and log in before building
llm_aggrefact_dataset = LLMAggreFactDataset(name="llm_aggrefact", hf_path="lytang/LLM-AggreFact")
