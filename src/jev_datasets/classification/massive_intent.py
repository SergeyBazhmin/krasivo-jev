from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which intent does this request to a virtual assistant express?"


class MassiveIntentDataset(JevDataset):
    def prepare(self):
        # `label` is the intent name itself, e.g. "alarm_set"
        ids = sorted({id for rows in self.data.values() for id in rows["label"]})
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, options, x["label"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


massive_intent_dataset = MassiveIntentDataset(
    name="massive_intent", hf_path="mteb/amazon_massive_intent", hf_name="en"
)
