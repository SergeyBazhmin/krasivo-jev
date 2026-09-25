from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, slugify

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
# `type` is missing on ~13k tickets, which are dropped
TYPES = ["Incident", "Request", "Problem", "Change"]
PRIORITIES = ["very_low", "low", "medium", "high", "critical"]


class TicketDataset(JevDataset):
    """One label column of the helpdesk tickets, predicted from subject and body."""

    def __init__(self, name: str, column: str, values: list[str], question: str):
        super().__init__(name=name, hf_path=HF_PATH)
        self.column = column
        self.ids = {value: slugify(value) for value in values}
        self.options = make_options(list(self.ids.values()), [value.replace("_", " ") for value in values])
        self.question = question

    def prepare(self):
        self.data = self.data.filter(lambda x: x[self.column] in self.ids)
        self.data = self.data.map(
            lambda x: make_sample(
                "\n\n".join(part for part in (x["subject"], x["body"]) if part),
                self.question,
                self.options,
                self.ids[x[self.column]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ticket_routing_dataset = TicketDataset(
    name="ticket_routing",
    column="queue",
    values=QUEUES,
    question="Which support queue should this ticket be routed to?",
)
ticket_type_dataset = TicketDataset(
    name="ticket_type", column="type", values=TYPES, question="What type of support ticket is this?"
)
ticket_priority_dataset = TicketDataset(
    name="ticket_priority",
    column="priority",
    values=PRIORITIES,
    question="What priority should this support ticket get?",
)
