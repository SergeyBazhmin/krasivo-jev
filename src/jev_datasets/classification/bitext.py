from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class BitextDataset(JevDataset):
    """Bitext's synthetic support chatbots: `instruction` is the customer message, `intent` a snake_case id."""

    def __init__(self, name: str, hf_path: str, question: str):
        super().__init__(name=name, hf_path=hf_path)
        self.question = question

    def prepare(self):
        # `intent` is a plain string column, so the option set is read off the data
        ids = sorted({id for rows in self.data.values() for id in rows["intent"]})
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(x["instruction"], self.question, options, x["intent"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


bitext_customer_support_dataset = BitextDataset(
    name="bitext_customer_support",
    hf_path="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    question="Which customer service intent does this message express?",
)
bitext_retail_banking_dataset = BitextDataset(
    name="bitext_retail_banking",
    hf_path="bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
    question="Which banking intent does this customer message express?",
)
bitext_insurance_dataset = BitextDataset(
    name="bitext_insurance",
    hf_path="bitext/Bitext-insurance-llm-chatbot-training-dataset",
    question="Which insurance intent does this customer message express?",
)
bitext_telco_dataset = BitextDataset(
    name="bitext_telco",
    hf_path="bitext/Bitext-telco-llm-chatbot-training-dataset",
    question="Which telecom intent does this customer message express?",
)
