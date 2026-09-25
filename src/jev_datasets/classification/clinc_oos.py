from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which intent does this user request express?"


class ClincOOSDataset(JevDataset):
    def prepare(self):
        # 150 in-scope intents over 10 domains (banking, credit cards, travel, ...) plus "oos"
        ids = self.class_names("intent")
        texts = ["out of scope" if id == "oos" else id.replace("_", " ") for id in ids]
        options = make_options(ids, texts)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, options, ids[x["intent"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


clinc_oos_dataset = ClincOOSDataset(name="clinc_oos", hf_path="clinc/clinc_oos", hf_name="plus")
