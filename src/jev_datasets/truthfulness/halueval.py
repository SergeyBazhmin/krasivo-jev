from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Does it contain hallucinated information?",
    "Does it include made-up or unsupported claims?",
    "Is any of it hallucinated?",
    "Does it state anything that isn't backed by the facts?",
    "Is it unfaithful to the given information?",
]
OPTIONS = make_options(["yes", "no"])


def paired_samples(state: str, kind: str, right: str, hallucinated: str) -> list[dict]:
    """Each source row pairs a faithful and a hallucinated text; judge each on its own."""
    return [
        make_sample(state, f"{kind}: {right}\n{pick_question(QUESTIONS, right)}", OPTIONS, "no"),
        make_sample(state, f"{kind}: {hallucinated}\n{pick_question(QUESTIONS, hallucinated)}", OPTIONS, "yes"),
    ]


ABOUT = {
    "qa": [
        (
            "You are checking an AI model's answers for hallucinations. You are given knowledge and a question "
            "whose answer combines facts from it, followed by the model's answer. The answer is hallucinated when "
            "the knowledge does not support it or contradicts it."
        ),
        (
            "You check an AI model's answers for hallucinations. Below are some knowledge and a question whose "
            "answer combines facts from it, then the model's answer. The answer is hallucinated if the knowledge "
            "does not support it or contradicts it."
        ),
        (
            "Below are a piece of knowledge, a question whose answer draws on facts from it, and an AI model's "
            "answer. Judge whether the answer is hallucinated, meaning the knowledge does not support it or "
            "contradicts it."
        ),
        (
            "As a reviewer of an AI model's answers, you look for hallucinations. You get knowledge and a question "
            "that combines facts from it, followed by the model's answer; the answer is hallucinated when it is "
            "unsupported by the knowledge or contradicts it."
        ),
        (
            "An AI model answered a question whose answer combines facts from the given knowledge. Its answer "
            "follows. The answer counts as hallucinated when the knowledge does not support it or contradicts it."
        ),
    ],
    "dialogue": [
        (
            "You are checking an AI assistant's responses for hallucinations. You are given facts about movies, "
            "books, music or sports, a chat between a human and the assistant, and the assistant's next response. "
            "The response is hallucinated when it states something the facts and the chat do not support, or that "
            "contradicts them."
        ),
        (
            "You check an AI assistant's responses for hallucinations. Below are facts about movies, books, music "
            "or sports, a chat between a human and the assistant, and the assistant's next response. The response "
            "is hallucinated if it states something the facts and the chat do not support, or that contradicts "
            "them."
        ),
        (
            "Below are some facts about movies, books, music or sports, a chat between a human and an AI "
            "assistant, and the assistant's next response. Judge whether the response is hallucinated, that is, "
            "whether it says something the facts and the chat do not support or that contradicts them."
        ),
        (
            "As a reviewer of an AI assistant's responses, you look for hallucinations. You get facts about "
            "movies, books, music or sports, a human's chat with the assistant, and its next response, which is "
            "hallucinated when it states something unsupported by the facts and the chat, or contradicting them."
        ),
        (
            "A human chatted with an AI assistant, and facts about movies, books, music or sports are given along "
            "with the assistant's next response. The response counts as hallucinated when it states something the "
            "facts and the chat do not support, or that contradicts them."
        ),
    ],
    "summarization": [
        (
            "You are checking AI-written summaries for hallucinations. You are given a news article and a summary "
            "of it. The summary is hallucinated when it states something the article does not support, or that "
            "contradicts it."
        ),
        (
            "You check summaries written by AI for hallucinations. Below are a news article and a summary of it. "
            "The summary is hallucinated if it states something the article does not support, or that contradicts "
            "it."
        ),
        (
            "Below is a news article, followed by an AI-written summary of it. Judge whether the summary is "
            "hallucinated, that is, whether it says something the article does not support or that contradicts it."
        ),
        (
            "As a reviewer of AI-written summaries, you look for hallucinations. Given a news article and its "
            "summary, the summary is hallucinated when it states something unsupported by the article or "
            "contradicting it."
        ),
        (
            "An AI summarized the news article below. The summary counts as hallucinated when it states something "
            "the article does not support, or that contradicts it."
        ),
    ],
}
# "general" is left out: its ChatGPT responses come with no reference knowledge, so
# spotting a hallucination takes world knowledge
SAMPLES = {
    "qa": lambda x: paired_samples(
        with_context(ABOUT["qa"], f"Knowledge: {x['knowledge']}\n\nQuestion: {x['question']}"),
        "Answer",
        x["right_answer"],
        x["hallucinated_answer"],
    ),
    "dialogue": lambda x: paired_samples(
        with_context(ABOUT["dialogue"], f"Knowledge: {x['knowledge']}\n\nChat: {x['dialogue_history']}"),
        "Response",
        x["right_response"],
        x["hallucinated_response"],
    ),
    "summarization": lambda x: paired_samples(
        with_context(ABOUT["summarization"], x["document"], "Article"),
        "Summary",
        x["right_summary"],
        x["hallucinated_summary"],
    ),
}


class HaluEvalDataset(JevDataset):
    type = DatasetType.NOUL

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
