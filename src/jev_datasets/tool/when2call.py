import re

from datasets import DatasetDict, concatenate_datasets, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    make_options,
    make_sample,
    pick_question,
    stable_shuffle,
    with_context,
)

QUESTIONS = [
    "Which reply should the assistant give next?",
    "What should the assistant answer?",
    "Pick the assistant's correct next reply.",
    "Which response is right for the assistant here?",
    "How should the assistant respond?",
]
# When2Call's own convention for when a tool may be called, so the state carries it
ABOUT = [
    (
        "You decide how an AI assistant with tools responds to a user. It calls a tool only when one of the given "
        "tools fits the request and every required argument is known; it asks a follow-up question when a fitting "
        "tool lacks a required argument; and it says it cannot help when no tool fits a request that needs live data "
        "or an action. It never makes up results it has not got from a tool."
    ),
    (
        "Below are the tools an AI assistant can use and a conversation with a user. The assistant may call a tool "
        "only if one fits the request and all its required arguments are known. If a fitting tool is missing a "
        "required argument, it asks the user for it; if no tool fits a request that needs real-time data or an "
        "action, it admits it cannot do it. It never invents tool results."
    ),
    (
        "You judge the replies of a tool-using AI assistant. A tool call is right only when a given tool fits and "
        "all of its required arguments are known. When a required argument is missing, the right reply asks for "
        "it. When no tool fits a request for live data or an action, the right reply says the assistant cannot "
        "help. Answers that pretend to have tool results are wrong."
    ),
    (
        "An AI assistant has the tools listed below. Rules: call a tool when one fits and its required arguments "
        "are all given; ask a follow-up question when a fitting tool still needs a required argument; say it "
        "cannot help when no tool fits and the request needs live data or an action; never fabricate results."
    ),
    (
        "As the reviewer of an AI assistant with tools, choose the reply it should give. It uses a tool only when "
        "one matches the request and every required argument is known, asks for whatever required argument is "
        "missing, and declines when none of its tools can serve a request for live data or an action. It must not "
        "make up tool results."
    ),
]
TOOL_CALL = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)


def render_reply(text: str) -> str:
    return TOOL_CALL.sub(lambda m: f"Tool call: {m[1].strip()}", text).strip()


def render_context(tools: list[str], messages: list[dict]) -> str:
    tools = "\n".join(tools) if tools else "(no tools)"
    dialogue = "\n".join(f"{m['role'].capitalize()}: {render_reply(m['content'])}" for m in messages)
    return with_context(ABOUT, f"Tools:\n{tools}\n\nConversation:\n{dialogue}")


def make_pref_sample(x: dict) -> dict:
    state = render_context(x["tools"], x["messages"])
    chosen = render_reply(x["chosen_response"]["content"])
    texts = stable_shuffle([chosen, render_reply(x["rejected_response"]["content"])], state)
    ids = ["A", "B"]
    return make_sample(state, pick_question(QUESTIONS, state), make_options(ids, texts), ids[texts.index(chosen)])


def make_mcq_sample(x: dict) -> dict:
    state = render_context(x["tools"], [{"role": "user", "content": x["question"]}])
    answers = dict(x["answers"])
    # the tool call is stored as one bare call, the train replies as a list of calls
    answers["tool_call"] = f"Tool call: [{answers['tool_call']}]"
    ids = stable_shuffle(list(answers), state)
    return make_sample(
        state, pick_question(QUESTIONS, state), make_options(ids, [answers[i] for i in ids]), x["correct_answer"]
    )


class When2CallDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # The SFT config has only the gold reply; the preference config pairs it with a wrong one. The 4-way MCQ
        # (`llm_judge` is a subset of it) is trained on too: both go to train and `regroup` carves validation/test.
        return DatasetDict(
            {
                "pref": load_dataset(self.hf_path, "train_pref", split="train"),
                "mcq": load_dataset(self.hf_path, "test", split="mcq"),
            }
        )

    def prepare(self):
        pref = self.data["pref"].filter(
            lambda x: render_reply(x["chosen_response"]["content"]) != render_reply(x["rejected_response"]["content"])
        )
        mcq = self.data["mcq"]
        self.data = DatasetDict(
            {
                "train": concatenate_datasets(
                    [
                        pref.map(make_pref_sample, remove_columns=pref.column_names, features=SAMPLE_FEATURES),
                        mcq.map(make_mcq_sample, remove_columns=mcq.column_names, features=SAMPLE_FEATURES),
                    ]
                )
            }
        )


when2call_dataset = When2CallDataset(name="when2call", hf_path="nvidia/When2Call")
