import hashlib
from collections import defaultdict

from datasets import DatasetDict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Which of the listed emotions is most clearly expressed in this message?",
    "Which emotion category best matches the emotional tone of this text?",
    "What is the main emotion conveyed by this message?",
    "Which of these feelings best describes the emotion expressed in the text?",
    "Which listed emotion best fits this message, based on its wording?",
]
ABOUT = (
    "Classify the emotion expressed in the message using the six categories below. Base the answer on the text, "
    "including negation and who experiences the emotion. A mentioned emotion is not necessarily expressed by the "
    "writer. Choose the best matching category; do not infer the writer's actual or current mental state."
)
ANSWERS = {
    "sadness": "Sorrow, unhappiness, grief, disappointment or discouragement.",
    "joy": "Happiness, pleasure, satisfaction or positive excitement.",
    "love": "Affection, caring, tenderness or attachment.",
    "anger": "Anger, irritation, frustration or resentment.",
    "fear": "Fear, anxiety, apprehension or feeling threatened or insecure.",
    "surprise": "Surprise, astonishment or amazement at something unexpected.",
}
# Reviewed copies of validation/test messages: contraction spacing, a truncated ending,
# and "can"/"could" in an otherwise identical passage. Hashes use casefold + collapsed whitespace.
# Evidence and full text pairs: docs/dataset-reviews/emotion.md, reviewed-near-duplicates.json.
TRAIN_HELDOUT_DUPLICATES = frozenset(
    {
        "ed821f6ddd7e0c7e6c05043abcb795e85c3a0cec6811f2d208711524cdc79705",
        "d5805bf2374f134d5e97f66babc23d1a3afa42a60962df40cf51807dda0f668f",
        "861e307ed193ddcbf0d1a60f38d16c662239517985798baac4144210281029e9",
    }
)


class EmotionDataset(JevDataset):
    type = DatasetType.CHOICE

    def clean_source(self) -> None:
        data = self.data
        names = self.class_names()
        labels: dict[str, set[int]] = defaultdict(set)
        keys = {}
        for split, rows in data.items():
            keys[split] = [" ".join(text.casefold().split()) for text in rows["text"]]
            for key, label in zip(keys[split], rows["label"]):
                if key and 0 <= label < len(names):
                    labels[key].add(label)

        # Conflicting source labels do not establish a unique answer (or annotator vote shares).
        # Drop every occurrence; keep consistent repeats once, preferring held-out splits.
        priority = {"test": 0, "validation": 1, "dev": 1, "val": 1, "train": 2}
        seen: set[str] = set()
        cleaned = {}
        for split in sorted(data, key=lambda name: (priority.get(name, 2), name)):
            indices = []
            for index, (key, label) in enumerate(zip(keys[split], data[split]["label"])):
                if split == "train" and hashlib.sha256(key.encode()).hexdigest() in TRAIN_HELDOUT_DUPLICATES:
                    continue
                if key and 0 <= label < len(names) and len(labels[key]) == 1 and key not in seen:
                    indices.append(index)
                    seen.add(key)
            cleaned[split] = data[split].select(indices)
        self.data = DatasetDict({split: cleaned[split] for split in data})

    def prepare(self):
        self.clean_source()
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Message", ANSWERS),
                pick_question(QUESTIONS, x["text"]),
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


emotion_dataset = EmotionDataset(name="emotion", hf_path="dair-ai/emotion", hf_name="split")
