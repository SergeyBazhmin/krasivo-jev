from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, slugify

QUESTION = "Which support queue should this ticket be routed to?"

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
QUEUE_IDS = {queue: slugify(queue) for queue in QUEUES}
OPTIONS = make_options(list(QUEUE_IDS.values()), QUEUES)


class TicketRoutingDataset(JevDataset):
    def prepare(self):
        self.data = self.data.filter(lambda x: x["queue"] in QUEUE_IDS)
        self.data = self.data.map(
            lambda x: make_sample(
                "\n\n".join(part for part in (x["subject"], x["body"]) if part),
                QUESTION,
                OPTIONS,
                QUEUE_IDS[x["queue"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ticket_routing_dataset = TicketRoutingDataset(
    name="ticket_routing", hf_path="Tobi-Bueck/customer-support-tickets"
)
