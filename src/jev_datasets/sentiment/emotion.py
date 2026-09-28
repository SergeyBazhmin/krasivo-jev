from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Which emotion does the author of this message express?",
    "How is the author of this message feeling?",
    "What emotion is conveyed in this text?",
    "Which feeling best describes the writer's mood?",
    "What is the dominant emotion in this message?",
]
ABOUT = (
    "You are given a Twitter message in which the author describes how they feel. Decide which of six basic emotions "
    "it expresses: sadness, joy, love, anger, fear or surprise."
)


class EmotionDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # sadness, joy, love, anger, fear, surprise
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Message"),
                pick_question(QUESTIONS, x["text"]),
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


emotion_dataset = EmotionDataset(name="emotion", hf_path="dair-ai/emotion", hf_name="split")
