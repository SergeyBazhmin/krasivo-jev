from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which banking intent does this customer message express?"


class Banking77Dataset(JevDataset):
    def prepare(self):
        # intents are snake_case ids, e.g. "card_payment_fee_charged"
        ids = self.class_names()
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, options, ids[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


banking77_dataset = Banking77Dataset(name="banking77", hf_path="legacy-datasets/banking77")
