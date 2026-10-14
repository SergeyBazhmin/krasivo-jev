from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Which intent does this user request express?",
    "What does the user want to do?",
    "What is the intent behind this request?",
    "Which intent best matches this user query?",
    "Classify the intent of this message, or mark it out of scope.",
]
ABOUT = [
    (
        "You are a task-oriented virtual assistant. A user has made the request below. Decide which of your 150 "
        "supported intents it expresses. They span 10 domains: banking, credit cards, kitchen and dining, home, "
        "auto and commute, travel, utility, work, small talk, and meta (talking about the assistant itself). If "
        "the request fits none of them, it is out of scope."
    ),
    (
        "You are a virtual assistant that handles tasks. Decide which of your 150 supported intents the user's "
        "request below expresses. The intents span 10 domains: banking, credit cards, kitchen and dining, home, "
        "auto and commute, travel, utility, work, small talk, and meta (about the assistant itself). A request "
        "that fits none of them is out of scope."
    ),
    (
        "Below is a request a user made to a task-oriented virtual assistant that supports 150 intents across 10 "
        "domains: banking, credit cards, kitchen and dining, home, auto and commute, travel, utility, work, small "
        "talk, and meta (talking about the assistant itself). Decide which intent it expresses, or that it is out "
        "of scope if it fits none."
    ),
    (
        "As a task-oriented virtual assistant, you support 150 intents in 10 domains (banking, credit cards, "
        "kitchen and dining, home, auto and commute, travel, utility, work, small talk, and meta, meaning talk "
        "about the assistant itself). Classify the user's request below into one of them, or as out of scope when "
        "none fits."
    ),
    (
        "A user has made the request below to a task-oriented virtual assistant. Decide which of its 150 intents "
        "the request expresses; they cover 10 domains: banking, credit cards, kitchen and dining, home, auto and "
        "commute, travel, utility, work, small talk and meta (talking about the assistant itself). Requests "
        "outside all of them are out of scope."
    ),
]


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
