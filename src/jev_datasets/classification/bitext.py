from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question


class BitextDataset(JevDataset):
    type = DatasetType.CHOICE

    """Bitext's synthetic support chatbots: `instruction` is the customer message, `intent` a snake_case id."""

    def __init__(self, name: str, hf_path: str, questions: list[str]):
        super().__init__(name=name, hf_path=hf_path)
        self.questions = questions

    def prepare(self):
        # `intent` is a plain string column, so the option set is read off the data
        ids = sorted({id for rows in self.data.values() for id in rows["intent"]})
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(
                x["instruction"], pick_question(self.questions, x["instruction"]), options, x["intent"]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


bitext_customer_support_dataset = BitextDataset(
    name="bitext_customer_support",
    hf_path="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    questions=[
        "Which customer service intent does this message express?",
        "What does this customer service customer want?",
        "What is the customer's intent in this message?",
        "Which customer service request is the customer making?",
        "How should a customer service chatbot classify this message?",
    ],
)
bitext_retail_banking_dataset = BitextDataset(
    name="bitext_retail_banking",
    hf_path="bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
    questions=[
        "Which banking intent does this customer message express?",
        "What does this banking customer want?",
        "What is the customer's intent in this message?",
        "Which banking request is the customer making?",
        "How should a banking chatbot classify this message?",
    ],
)
bitext_insurance_dataset = BitextDataset(
    name="bitext_insurance",
    hf_path="bitext/Bitext-insurance-llm-chatbot-training-dataset",
    questions=[
        "Which insurance intent does this customer message express?",
        "What does this insurance customer want?",
        "What is the customer's intent in this message?",
        "Which insurance request is the customer making?",
        "How should a insurance chatbot classify this message?",
    ],
)
bitext_telco_dataset = BitextDataset(
    name="bitext_telco",
    hf_path="bitext/Bitext-telco-llm-chatbot-training-dataset",
    questions=[
        "Which telecom intent does this customer message express?",
        "What does this telecom customer want?",
        "What is the customer's intent in this message?",
        "Which telecom request is the customer making?",
        "How should a telecom chatbot classify this message?",
    ],
)
