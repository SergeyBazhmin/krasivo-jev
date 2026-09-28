from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Which intent does this user request express?",
    "What does the user want to do?",
    "What is the intent behind this request?",
    "Which intent best matches this user query?",
    "Classify the intent of this message, or mark it out of scope.",
]
ABOUT = (
    "You are a task-oriented virtual assistant. A user has made the request below. Decide which of your 150 supported "
    "intents it expresses. They span 10 domains: banking, credit cards, kitchen and dining, home, auto and commute, "
    "travel, utility, work, small talk, and meta (talking about the assistant itself). If the request fits none of "
    "them, it is out of scope."
)


class ClincOOSDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # 150 in-scope intents over 10 domains (banking, credit cards, travel, ...) plus "oos"
        ids = self.class_names("intent")
        texts = ["out of scope" if id == "oos" else id.replace("_", " ") for id in ids]
        options = make_options(ids, texts)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Request"),
                pick_question(QUESTIONS, x["text"]),
                options,
                ids[x["intent"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


clinc_oos_dataset = ClincOOSDataset(name="clinc_oos", hf_path="clinc/clinc_oos", hf_name="plus")
