from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Which banking intent does this customer message express?",
    "What does this bank customer want?",
    "What is the customer's intent in this banking query?",
    "Which banking issue is this customer writing about?",
    "How should this bank support message be classified?",
]
ABOUT = [
    (
        "You are the support assistant of an online bank. A customer has sent the message below. Decide which of "
        "77 fine-grained intents it expresses. Many differ only in detail (a card payment that was declined, is "
        "still pending or was charged twice), so pay attention to exactly what the customer describes."
    ),
    (
        "You work in customer support for an online bank, and a customer has written the message below. Decide "
        "which of 77 fine-grained intents it expresses. Many intents differ only in detail (a card payment that "
        "was declined, is still pending, or was charged twice), so read exactly what the customer describes."
    ),
    (
        "Below is a message a customer sent to an online bank's support. Classify it into one of 77 fine-grained "
        "intents. Pay close attention to what exactly the customer describes, since many intents differ only in "
        "detail: a card payment can be declined, still pending, or charged twice."
    ),
    (
        "As the support assistant of an online bank, you sort customer messages into 77 fine-grained intents. Read "
        "the message below carefully: many intents differ only in a detail, such as whether a card payment was "
        "declined, is still pending or was charged twice."
    ),
    (
        "An online bank's customer sent the message below to support. Decide which of 77 detailed intents it "
        "expresses, attending to exactly what is described, because intents can differ only in detail (for example "
        "a card payment declined, pending or charged twice)."
    ),
]


class Banking77Dataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # intents are snake_case ids, e.g. "card_payment_fee_charged"
        ids = self.class_names()
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Message"), pick_question(QUESTIONS, x["text"]), options, ids[x["label"]]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


banking77_dataset = Banking77Dataset(name="banking77", hf_path="legacy-datasets/banking77")
