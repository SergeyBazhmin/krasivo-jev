from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify

HF_PATH = "Tobi-Bueck/customer-support-tickets"

# The `queue` column mixes the helpdesk queues below (~49k tickets) with a long tail of
# "Category/Subcategory" topic tags (~300 tickets each) that are not routing targets.
QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]
# `type` is missing on ~13k tickets, which are dropped; `priority` is left out, being a
# business call that the ticket text doesn't settle
TYPES = ["Incident", "Request", "Problem", "Change"]


class TicketDataset(JevDataset):
    type = DatasetType.CHOICE

    """One label column of the helpdesk tickets, predicted from subject and body."""

    def __init__(self, name: str, column: str, values: list[str], questions: list[str]):
        super().__init__(name=name, hf_path=HF_PATH)
        self.column = column
        self.ids = {value: slugify(value) for value in values}
        self.options = make_options(list(self.ids.values()), [value.replace("_", " ") for value in values])
        self.questions = questions

    def prepare(self):
        self.data = self.data.filter(lambda x: x[self.column] in self.ids)
        self.data = self.data.map(self.to_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES)

    def to_sample(self, x: dict) -> dict:
        ticket = "\n\n".join(part for part in (x["subject"], x["body"]) if part)
        return make_sample(ticket, pick_question(self.questions, ticket), self.options, self.ids[x[self.column]])


ticket_routing_dataset = TicketDataset(
    name="ticket_routing",
    column="queue",
    values=QUEUES,
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
    values=TYPES,
    questions=[
        "What type of support ticket is this?",
        "Is this ticket an incident, a request, a problem or a change?",
        "How should this ticket be categorized?",
        "What kind of ticket did the customer open?",
        "Which ticket type fits this request?",
    ],
)
