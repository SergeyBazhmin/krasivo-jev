from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Which emotion does the author of this message express?",
    "How is the author of this message feeling?",
    "What emotion is conveyed in this text?",
    "Which feeling best describes the writer's mood?",
    "What is the dominant emotion in this message?",
]


class EmotionDataset(JevDataset):
    def prepare(self):
        # sadness, joy, love, anger, fear, surprise
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], pick_question(QUESTIONS, x["text"]), options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


emotion_dataset = EmotionDataset(name="emotion", hf_path="dair-ai/emotion", hf_name="split")
