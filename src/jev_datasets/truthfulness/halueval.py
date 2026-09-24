from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample

QUESTION = "Does it contain hallucinated information?"
OPTIONS = make_options(["yes", "no"])


def paired_samples(state: str, kind: str, right: str, hallucinated: str) -> list[dict]:
    """Each source row pairs a faithful and a hallucinated text; judge each on its own."""
    return [
        make_sample(state, f"{kind}: {right}\n{QUESTION}", OPTIONS, "no"),
        make_sample(state, f"{kind}: {hallucinated}\n{QUESTION}", OPTIONS, "yes"),
    ]


SAMPLES = {
    "qa": lambda x: paired_samples(
        f"{x['knowledge']}\n\nQuestion: {x['question']}", "Answer", x["right_answer"], x["hallucinated_answer"]
    ),
    "dialogue": lambda x: paired_samples(
        f"{x['knowledge']}\n\n{x['dialogue_history']}", "Response", x["right_response"], x["hallucinated_response"]
    ),
    "summarization": lambda x: paired_samples(
        x["document"], "Summary", x["right_summary"], x["hallucinated_summary"]
    ),
    # ChatGPT responses annotated as a whole
    "general": lambda x: [
        make_sample(x["user_query"], f"Response: {x['chatgpt_response']}\n{QUESTION}", OPTIONS, x["hallucination"])
    ],
}


class HaluEvalDataset(JevDataset):
    def __init__(self, task: str):
        super().__init__(name=f"halueval_{task}", hf_path="pminervini/HaluEval", hf_name=task)

    def prepare(self):
        self.data = self.data.map(
            explode(SAMPLES[self.hf_name]),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


halueval_qa_dataset = HaluEvalDataset("qa")
halueval_dialogue_dataset = HaluEvalDataset("dialogue")
halueval_summarization_dataset = HaluEvalDataset("summarization")
halueval_general_dataset = HaluEvalDataset("general")
