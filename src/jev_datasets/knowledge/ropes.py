from collections import defaultdict

from datasets import DatasetDict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, pick_about, text_choice

ABOUT = [
    (
        "You are given a background passage that explains a cause and effect, and a situation that applies it. "
        "Use the background to reason about the situation and answer the question."
    ),
    (
        "Below are a background passage describing how something works and a new situation. Apply what the "
        "background explains to the situation to answer the question."
    ),
    (
        "A background passage explains an effect, and a situation describes a new case. Reason from the "
        "background about the situation, then answer the question."
    ),
    (
        "You apply explanations to new cases. Read the background, which describes a cause and effect, and the "
        "situation, and answer the question about the situation."
    ),
    (
        "Read the background passage and the situation that follows. The background explains how something works; "
        "use it to answer the question about the situation."
    ),
]


def context_key(x: dict) -> tuple[str, str]:
    return x["background"], x["situation"]


def ropes_samples(siblings: dict[tuple[str, str], list[str]]):
    def samples(x: dict) -> list[dict]:
        answers = [text.strip() for text in x["answers"]["text"] if text.strip()]
        if not answers:
            return []
        # answers are spans with no options; the other questions on the same situation usually ask the opposite
        # ("higher" vs "lower"), so their answers are the natural distractors
        excluded = {text.casefold() for text in answers}
        distractors = [text for text in siblings[context_key(x)] if text.casefold() not in excluded]
        state = (
            f"{pick_about(ABOUT, x['background'] + x['situation'])}\n\n"
            f"Background:\n{x['background']}\n\nSituation:\n{x['situation']}"
        )
        sample = text_choice(state, x["question"], answers[0], distractors, x["id"])
        return [sample] if sample else []

    return samples


class ROPESDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        data = {}
        for split, rows in self.data.items():
            siblings = defaultdict(list)
            for x in rows:
                siblings[context_key(x)] += [text.strip() for text in x["answers"]["text"][:1] if text.strip()]
            data[split] = rows.map(
                explode(ropes_samples(siblings)),
                batched=True,
                remove_columns=self.source_columns,
                features=SAMPLE_FEATURES,
                load_from_cache_file=False,
            )
        self.data = DatasetDict(data)


ropes_dataset = ROPESDataset(name="ropes", hf_path="allenai/ropes", hf_name="plain_text")
