import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, map_records, record
from jev_datasets.utils import pick_question, stable_shuffle, text_choice

REVISION = "848cc8c05e0a04def79326539de827f1b28786a0"
BASE_URL = f"https://raw.githubusercontent.com/cylnlp/dialogsum/{REVISION}/DialogSum_Data"
SPEAKER_QUESTIONS = [
    'Who says this line: "{quote}"?',
    'Which participant says "{quote}"?',
    'Identify the speaker of "{quote}".',
    'Who is the author of the utterance "{quote}"?',
    'Choose the participant who says "{quote}" in this dialogue.',
]
NEXT_QUESTIONS = [
    'Which utterance immediately follows "{quote}" in the recorded dialogue?',
    'What is the next recorded line after "{quote}"?',
    'Select the line that comes directly after "{quote}".',
    'Find the utterance immediately after "{quote}" in the transcript.',
    'According to the transcript order, what is said next after "{quote}"?',
]
TURN = re.compile(r"^(#Person\d+#):\s*(.+)$")


def dialogsum_records(paths: dict[str, str]) -> Iterator[dict]:
    for split, path in paths.items():
        for line in Path(path).read_text().splitlines():
            text = json.loads(line)["dialogue"]
            # Summaries and topics are not evidence and never enter a sample or its target.
            yield {**record({"text": text}, content_group(text.strip())), "split": split}


def dialogsum_samples(row: dict) -> list[dict]:
    text = json.loads(row["payload"])["text"]
    if not 40 <= len(text) <= 12000:
        return []
    matches = [TURN.fullmatch(line.strip()) for line in text.splitlines() if line.strip()]
    if not matches or any(match is None for match in matches):
        return []
    turns = [(match[1], match[2].strip()) for match in matches]
    speakers = sorted({speaker for speaker, _ in turns})
    if len(speakers) < 2:
        return []
    occurrences = defaultdict(list)
    for index, (_, utterance) in enumerate(turns):
        occurrences[utterance.casefold()].append(index)
    state = "Recorded dialogue. Use the speaker tags and the actual transcript order.\n\n" + text
    samples = []
    eligible = [index for index, (_, utterance) in enumerate(turns) if len(utterance) >= 20]
    for index in stable_shuffle(eligible, row["group"]):
        speaker, quote = turns[index]
        # Repeated lines such as 'Yes' can have several speakers and several different successors.
        if len(occurrences[quote.casefold()]) != 1:
            continue
        seed = text + str(index)
        question = pick_question(SPEAKER_QUESTIONS, seed).format(quote=quote)
        samples.append(text_choice(state, question, speaker, [value for value in speakers if value != speaker], seed))
        if index + 1 < len(turns):
            answer = turns[index + 1][1]
            candidates = [
                utterance for _, utterance in turns if utterance.casefold() not in {answer.casefold(), quote.casefold()}
            ]
            question = pick_question(NEXT_QUESTIONS, seed).format(quote=quote)
            if sample := text_choice(state, question, answer, candidates, seed + ":next"):
                samples.append(sample)
        if len(samples) >= 4:
            break
    return samples[:4]


class DialogSumTurnsDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        paths = DownloadManager().download(
            {
                split: f"{BASE_URL}/dialogsum.{name}.jsonl"
                for split, name in (("train", "train"), ("validation", "dev"), ("test", "test"))
            }
        )
        return grouped_records(dialogsum_records, paths=paths)

    def prepare(self):
        self.data = map_records(self.data, dialogsum_samples, __file__)


dialogsum_turns_dataset = DialogSumTurnsDataset(name="dialogsum_turns", hf_path="cylnlp/dialogsum")
