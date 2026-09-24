from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, stable_shuffle

QUESTION = "Which assistant reply is better?"
REPLY_MARKER = "\n\nAssistant:"


def split_last_reply(transcript: str) -> tuple[str, str]:
    """(conversation so far, final assistant reply)"""
    cut = transcript.rfind(REPLY_MARKER)
    return transcript[:cut].strip(), transcript[cut + len(REPLY_MARKER):].strip()


class HHRLHFDataset(JevDataset):
    def prepare(self):
        # both transcripts share the conversation and differ only in the final reply
        self.data = self.data.filter(
            lambda x: split_last_reply(x["chosen"])[0] == split_last_reply(x["rejected"])[0]
        )
        self.data = self.data.map(self.to_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES)

    @staticmethod
    def to_sample(x: dict) -> dict:
        conversation, chosen = split_last_reply(x["chosen"])
        _, rejected = split_last_reply(x["rejected"])
        # the source always lists `chosen` first
        replies = stable_shuffle([("chosen", chosen), ("rejected", rejected)], conversation)
        return make_sample(
            conversation,
            QUESTION,
            make_options([id for id, _ in replies], [text for _, text in replies]),
            "chosen",
        )


hh_rlhf_dataset = HHRLHFDataset(name="hh_rlhf", hf_path="Anthropic/hh-rlhf")
