from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context


class BitextDataset(JevDataset):
    type = DatasetType.CHOICE

    """Bitext's synthetic support chatbots: `instruction` is the customer message, `intent` a snake_case id."""

    def __init__(self, name: str, hf_path: str, about: list[str], questions: list[str]):
        super().__init__(name=name, hf_path=hf_path)
        self.about = about
        self.questions = questions

    def prepare(self):
        # `intent` is a plain string column, so the option set is read off the data
        ids = sorted({id for rows in self.data.values() for id in rows["intent"]})
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(self.about, x["instruction"], "Message"),
                pick_question(self.questions, x["instruction"]),
                options,
                x["intent"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


bitext_customer_support_dataset = BitextDataset(
    name="bitext_customer_support",
    hf_path="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    about=[
        (
            "You are the customer service chatbot of a company. A customer has sent the message below. Decide "
            "which request it makes; the intents cover orders, shipping, payments, invoices, refunds, "
            "cancellations, accounts, feedback, and reaching a human agent."
        ),
        (
            "You are the customer service chatbot of a company, and a customer has sent the message below. Decide "
            "which request it makes. The intents cover orders, shipping, payments, invoices, refunds, "
            "cancellations, accounts, feedback, and reaching a human agent."
        ),
        (
            "Below is a message from a customer to the customer service chatbot of a company. Classify the request "
            "it makes; the intents cover orders, shipping, payments, invoices, refunds, cancellations, accounts, "
            "feedback, and reaching a human agent."
        ),
        (
            "As the customer service chatbot of a company, you route customer messages by intent. The intents "
            "cover orders, shipping, payments, invoices, refunds, cancellations, accounts, feedback, and reaching "
            "a human agent. Decide which request the message below makes."
        ),
        (
            "A customer wrote the message below to the customer service chatbot of a company. Decide what they are "
            "requesting; the possible intents cover orders, shipping, payments, invoices, refunds, cancellations, "
            "accounts, feedback, and reaching a human agent."
        ),
    ],
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
    about=[
        (
            "You are the virtual assistant of a retail bank. A customer has sent the message below. Decide which "
            "request it makes; the intents cover cards, accounts, transfers, loans, mortgages, fees, ATMs and "
            "branches, passwords, and reaching customer service or a human agent."
        ),
        (
            "You are the virtual assistant of a retail bank, and a customer has sent the message below. Decide "
            "which request it makes. The intents cover cards, accounts, transfers, loans, mortgages, fees, ATMs "
            "and branches, passwords, and reaching customer service or a human agent."
        ),
        (
            "Below is a message from a customer to the virtual assistant of a retail bank. Classify the request it "
            "makes; the intents cover cards, accounts, transfers, loans, mortgages, fees, ATMs and branches, "
            "passwords, and reaching customer service or a human agent."
        ),
        (
            "As the virtual assistant of a retail bank, you route customer messages by intent. The intents cover "
            "cards, accounts, transfers, loans, mortgages, fees, ATMs and branches, passwords, and reaching "
            "customer service or a human agent. Decide which request the message below makes."
        ),
        (
            "A customer wrote the message below to the virtual assistant of a retail bank. Decide what they are "
            "requesting; the possible intents cover cards, accounts, transfers, loans, mortgages, fees, ATMs and "
            "branches, passwords, and reaching customer service or a human agent."
        ),
    ],
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
    about=[
        (
            "You are the virtual assistant of an insurance company. A customer has sent the message below. Decide "
            "which request it makes; the intents cover policies and coverage, quotes and rates, claims and "
            "settlements, payments and invoices, information on each type of insurance, and reaching an agent."
        ),
        (
            "You are the virtual assistant of an insurance company, and a customer has sent the message below. "
            "Decide which request it makes. The intents cover policies and coverage, quotes and rates, claims and "
            "settlements, payments and invoices, information on each type of insurance, and reaching an agent."
        ),
        (
            "Below is a message from a customer to the virtual assistant of an insurance company. Classify the "
            "request it makes; the intents cover policies and coverage, quotes and rates, claims and settlements, "
            "payments and invoices, information on each type of insurance, and reaching an agent."
        ),
        (
            "As the virtual assistant of an insurance company, you route customer messages by intent. The intents "
            "cover policies and coverage, quotes and rates, claims and settlements, payments and invoices, "
            "information on each type of insurance, and reaching an agent. Decide which request the message below "
            "makes."
        ),
        (
            "A customer wrote the message below to the virtual assistant of an insurance company. Decide what they "
            "are requesting; the possible intents cover policies and coverage, quotes and rates, claims and "
            "settlements, payments and invoices, information on each type of insurance, and reaching an agent."
        ),
    ],
    questions=[
        "Which insurance intent does this customer message express?",
        "What does this insurance customer want?",
        "What is the customer's intent in this message?",
        "Which insurance request is the customer making?",
        "How should an insurance chatbot classify this message?",
    ],
)
bitext_telco_dataset = BitextDataset(
    name="bitext_telco",
    hf_path="bitext/Bitext-telco-llm-chatbot-training-dataset",
    about=[
        (
            "You are the virtual assistant of a telecom operator. A customer has sent the message below. Decide "
            "which request it makes; the intents cover phone plans, phones and services, roaming, signal coverage, "
            "usage and data charges, internet installation, payments and invoices, and reaching an agent."
        ),
        (
            "You are the virtual assistant of a telecom operator, and a customer has sent the message below. "
            "Decide which request it makes. The intents cover phone plans, phones and services, roaming, signal "
            "coverage, usage and data charges, internet installation, payments and invoices, and reaching an "
            "agent."
        ),
        (
            "Below is a message from a customer to the virtual assistant of a telecom operator. Classify the "
            "request it makes; the intents cover phone plans, phones and services, roaming, signal coverage, usage "
            "and data charges, internet installation, payments and invoices, and reaching an agent."
        ),
        (
            "As the virtual assistant of a telecom operator, you route customer messages by intent. The intents "
            "cover phone plans, phones and services, roaming, signal coverage, usage and data charges, internet "
            "installation, payments and invoices, and reaching an agent. Decide which request the message below "
            "makes."
        ),
        (
            "A customer wrote the message below to the virtual assistant of a telecom operator. Decide what they "
            "are requesting; the possible intents cover phone plans, phones and services, roaming, signal "
            "coverage, usage and data charges, internet installation, payments and invoices, and reaching an "
            "agent."
        ),
    ],
    questions=[
        "Which telecom intent does this customer message express?",
        "What does this telecom customer want?",
        "What is the customer's intent in this message?",
        "Which telecom request is the customer making?",
        "How should a telecom chatbot classify this message?",
    ],
)
