from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Какое намерение у пользователя?"


class RuMassiveIntentDataset(JevDataset):
    def prepare(self):
        # 60 intents, given as strings rather than a ClassLabel
        intents = sorted(set(self.data["train"]["label"]))
        options = make_options(intents, [intent.replace("_", " ") for intent in intents])
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, options, x["label"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ru_massive_intent_dataset = RuMassiveIntentDataset(
    name="ru_massive_intent", hf_path="mteb/amazon_massive_intent", hf_name="ru"
)
