from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which emotion does the author of this message express?"


class EmotionDataset(JevDataset):
    def prepare(self):
        # sadness, joy, love, anger, fear, surprise
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


emotion_dataset = EmotionDataset(name="emotion", hf_path="dair-ai/emotion", hf_name="split")
