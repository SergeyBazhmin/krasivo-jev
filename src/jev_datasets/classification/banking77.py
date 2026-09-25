from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Which banking intent does this customer message express?",
    "What does this bank customer want?",
    "What is the customer's intent in this banking query?",
    "Which banking issue is this customer writing about?",
    "How should this bank support message be classified?",
]


class Banking77Dataset(JevDataset):
    def prepare(self):
        # intents are snake_case ids, e.g. "card_payment_fee_charged"
        ids = self.class_names()
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(x["text"], pick_question(QUESTIONS, x["text"]), options, ids[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


banking77_dataset = Banking77Dataset(name="banking77", hf_path="legacy-datasets/banking77")
