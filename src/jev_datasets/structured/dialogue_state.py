import json
import re
from collections import defaultdict
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, record
from jev_datasets.utils import SAMPLE_FEATURES, explode, pick_question, stable_shuffle, text_choice

QUESTIONS = [
    "What is the user's current value for {slot}?",
    "Which value is currently requested for {slot}?",
    "After the latest message, what value does the user want for {slot}?",
    "Choose the current value of {slot} from this conversation.",
    "What is the latest agreed value for {slot}?",
]
ABOUT = (
    "Read the conversation up to the last message. Track the user's current request for each service. "
    "Later corrections replace earlier values; an assistant's suggestion counts only if the user accepts it. "
    "Use only the conversation, not unstated database results. 'Centre' and 'center' mean the same thing."
)
SGD_REVISION = "e852981ae34990f4358979625854259302feaa78"
MULTIWOZ_REVISION = "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b"


@lru_cache(maxsize=32768)
def normalized(value: str) -> str:
    value = value.casefold().replace("centre", "center").strip()
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
    for number, word in enumerate(words):
        value = re.sub(r"\b" + word + r"\b", str(number), value)
    value = value.replace("a.m.", "am").replace("p.m.", "pm")
    if match := re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", value):
        hour, minute = int(match[1]), int(match[2] or "0")
        if 1 <= hour <= 12 and minute < 60:
            return f"time:{hour % 12 + (12 if match[3] == 'pm' else 0):02d}:{minute:02d}"
    if (match := re.fullmatch(r"(\d{1,2}):(\d{2})", value)) and int(match[1]) < 24 and int(match[2]) < 60:
        return f"time:{int(match[1]):02d}:{int(match[2]):02d}"
    return re.sub(r"\W+", "", value)


def usable(value: str) -> bool:
    return bool(value.strip()) and value.casefold() not in {"dontcare", "don't care", "none", "not mentioned", "?"}


def visible_value(value: str, history: str) -> bool:
    value = value.casefold().replace("centre", "center")
    history = history.casefold().replace("centre", "center")
    return bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", history))


def dialogue_records(files: dict) -> Iterator[dict]:
    for split, paths in files.items():
        schema = json.loads(Path(paths["schema"]).read_text())
        schema = {service["service_name"]: service for service in schema}
        for path in paths["dialogues"]:
            for dialogue in json.loads(Path(path).read_text()):
                text = [(turn["speaker"], turn["utterance"]) for turn in dialogue["turns"]]
                # DB results and gold goals are neither needed for loading nor allowed as model input.
                turns = [
                    {
                        "speaker": turn["speaker"],
                        "utterance": turn["utterance"],
                        "frames": [
                            {"service": frame["service"], "state": frame.get("state", {})} for frame in turn["frames"]
                        ],
                    }
                    for turn in dialogue["turns"]
                ]
                payload = {
                    "dialogue": {"turns": turns},
                    "schema": {name: schema[name] for name in dialogue["services"] if name in schema},
                }
                yield {**record(payload, content_group(text)), "split": split}


def current_slots(dialogue: dict) -> Iterator[tuple[int, str, str, list[str], str]]:
    history = []
    previous = {}
    for turn_index, turn in enumerate(dialogue["turns"]):
        history.append(f"{turn['speaker']}: {turn['utterance']}")
        if turn["speaker"] != "USER":
            continue
        for frame in turn["frames"]:
            service = frame["service"]
            for slot, values in frame.get("state", {}).get("slot_values", {}).items():
                key = service, slot
                values = [value for value in values if usable(value)]
                if not values or previous.get(key) == values:
                    continue
                previous[key] = values
                yield turn_index, service, slot, values, "\n".join(history)


class DialogueStateDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        multiwoz = self.hf_name == "multiwoz"
        root = (
            f"https://raw.githubusercontent.com/budzianowski/multiwoz/{MULTIWOZ_REVISION}/data/MultiWOZ_2.2"
            if multiwoz
            else f"https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/{SGD_REVISION}"
        )
        counts = {"train": 17, "dev": 2, "test": 2} if multiwoz else {"train": 127, "dev": 20, "test": 34}
        urls = {}
        for source_split, count in counts.items():
            split = "validation" if source_split == "dev" else source_split
            urls[split] = {
                "schema": f"{root}/schema.json" if multiwoz else f"{root}/{source_split}/schema.json",
                "dialogues": [f"{root}/{source_split}/dialogues_{i:03d}.json" for i in range(1, count + 1)],
            }
        return grouped_records(dialogue_records, files=DownloadManager().download(urls))

    def prepare(self):
        pools = defaultdict(set)
        for row in self.data["train"]:
            payload = json.loads(row["payload"])
            for _, service, slot, values, _ in current_slots(payload["dialogue"]):
                pools[service, slot].update(values)
        pools = {key: sorted(values) for key, values in pools.items()}

        def convert(row: dict) -> list[dict]:
            payload = json.loads(row["payload"])
            samples = []
            for index, service, slot, values, history in current_slots(payload["dialogue"]):
                schema = payload["schema"].get(service, {})
                definition = next((item for item in schema.get("slots", []) if item["name"] == slot), None)
                if definition is None:
                    continue
                # Some belief values are implicit DB facts or incorrect annotations. Require visible evidence.
                visible = [value for value in values if visible_value(value, history)]
                if not visible:
                    continue
                accepted = {normalized(value) for value in values}
                boolean = any(token in slot.lower() for token in ("parking", "internet", "has_", "serves_"))
                # An incidental 'yes' elsewhere in a dialogue is not evidence of parking/internet availability.
                if boolean:
                    continue
                candidates = [
                    value
                    for value in pools.get((service, slot), []) + definition.get("possible_values", [])
                    if usable(value) and normalized(value) not in accepted
                ]
                seed = row["group"] + str(index) + service + slot
                description = definition["description"]
                state = f"{ABOUT}\n\nService: {schema.get('description', service)}\n\n{history}"
                question = pick_question(QUESTIONS, seed).format(slot=description)
                if sample := text_choice(state, question, visible[0], candidates, seed):
                    samples.append(sample)
            # Bound repeated supervision from long dialogues while keeping all prefixes in the source split.
            return stable_shuffle(samples, row["group"])[:4]

        self.data = self.data.map(
            explode(convert), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


multiwoz_state_dataset = DialogueStateDataset(
    name="multiwoz_state", hf_path="budzianowski/multiwoz", hf_name="multiwoz"
)
sgd_state_dataset = DialogueStateDataset(
    name="sgd_state", hf_path="google-research-datasets/dstc8-schema-guided-dialogue", hf_name="sgd"
)
