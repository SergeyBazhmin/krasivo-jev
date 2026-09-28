from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify, with_context

HF_PATH = "Tobi-Bueck/customer-support-tickets"

ABOUT = (
    "You work at a company's helpdesk. You are given a support ticket (subject and email body) and decide how it "
    "should be labeled."
)
# The `queue` column mixes the helpdesk queues below (~49k tickets) with a long tail of
# "Category/Subcategory" topic tags (~300 tickets each) that are not routing targets.
QUEUE_ANSWERS = {
    "Technical Support": "technical issues and support requests",
    "Product Support": "problems with or questions about a specific product",
    "Customer Service": "customer inquiries and service requests",
    "IT Support": "internal IT and infrastructure issues",
    "Billing and Payments": "billing issues and payment processing",
    "Returns and Exchanges": "product returns and exchanges",
    "Service Outages and Maintenance": "services that are down, and planned maintenance",
    "Sales and Pre-Sales": "sales inquiries and questions before buying",
    "Human Resources": "employee inquiries and HR issues",
    "General Inquiry": "general questions that fit no specialised queue",
}
# `type` is missing on ~13k tickets, which are dropped; `priority` is left out, being a
# business call that the ticket text doesn't settle
TYPE_ANSWERS = {
    "Incident": "an unexpected issue that needs immediate attention",
    "Request": "a routine inquiry or service request",
    "Problem": "an underlying issue that causes repeated incidents",
    "Change": "a planned change or update",
}


class TicketDataset(JevDataset):
    type = DatasetType.CHOICE

    """One label column of the helpdesk tickets, predicted from subject and body."""

    def __init__(self, name: str, column: str, answers: dict[str, str], questions: list[str]):
        super().__init__(name=name, hf_path=HF_PATH)
        self.column = column
        self.answers = answers
        self.ids = {value: slugify(value) for value in answers}
        self.options = make_options(list(self.ids.values()), list(answers))
        self.questions = questions

    def prepare(self):
        self.data = self.data.filter(lambda x: x[self.column] in self.ids)
        self.data = self.data.map(self.to_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES)

    def to_sample(self, x: dict) -> dict:
        ticket = "\n\n".join(part for part in (x["subject"], x["body"]) if part)
        state = with_context(ABOUT, ticket, "Ticket", self.answers)
        return make_sample(state, pick_question(self.questions, ticket), self.options, self.ids[x[self.column]])


ticket_routing_dataset = TicketDataset(
    name="ticket_routing",
    column="queue",
    answers=QUEUE_ANSWERS,
    questions=[
        "Which support queue should this ticket be routed to?",
        "Which team should handle this support ticket?",
        "Which department is this ticket for?",
        "Where should this ticket be assigned?",
        "Which helpdesk queue does this request belong in?",
    ],
)
ticket_type_dataset = TicketDataset(
    name="ticket_type",
    column="type",
    answers=TYPE_ANSWERS,
    questions=[
        "What type of support ticket is this?",
        "Is this ticket an incident, a request, a problem or a change?",
        "How should this ticket be categorized?",
        "What kind of ticket did the customer open?",
        "Which ticket type fits this request?",
    ],
)
