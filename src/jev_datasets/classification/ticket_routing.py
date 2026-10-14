import hashlib
import random
from collections import Counter, defaultdict

from datasets import Dataset, DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    make_options,
    make_sample,
    pick_question,
    slugify,
    stable_shuffle,
    with_context,
)

HF_PATH = "Tobi-Bueck/customer-support-tickets"

ABOUT = [
    (
        "You work at a company's helpdesk. You are given a support ticket (subject and email body) and decide how "
        "it should be labeled."
    ),
    (
        "You are on a company's helpdesk team. Given a support ticket with its subject and email body, decide how "
        "it should be labeled."
    ),
    (
        "Below is a support ticket sent to a company's helpdesk, with its subject and email body. Decide which "
        "label it should get."
    ),
    (
        "As a helpdesk agent at a company, you label incoming support tickets. Read the ticket's subject and email "
        "body and decide how to label it."
    ),
    (
        "A support ticket (subject and email body) has arrived at a company's helpdesk. Decide how the ticket "
        "should be labeled."
    ),
]
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
SUBJECT_ABOUT = [
    (
        "You work at a company's helpdesk. You are given the subject line of a support email, and each option is "
        "the body of a customer's email. Pick the email the subject was written for: its body covers what the "
        "subject names. Choose 'None of these emails' if none of them fits the subject."
    ),
    (
        "You are on a company's helpdesk team. Below is the subject line of a support email; each option is the "
        "body of a customer email. Pick the email this subject was written for, the one whose body covers what the "
        "subject names. If none fits the subject, choose 'None of these emails'."
    ),
    (
        "Given the subject line of a support email sent to a company's helpdesk, find its body among the options, "
        "each of which is a customer's email body. The right one covers what the subject names; choose 'None of "
        "these emails' when no body fits."
    ),
    (
        "As a helpdesk agent, you match subject lines to support emails. Each option is a customer email body. "
        "Choose the one the subject below was written for, whose body is about what the subject names, or 'None of "
        "these emails' if none of them fits."
    ),
    (
        "A company's helpdesk received support emails whose bodies are the options. Below is one email's subject "
        "line. Pick the email it belongs to (its body covers what the subject names), or 'None of these emails' if "
        "no option fits the subject."
    ),
]
SUBJECT_QUESTIONS = [
    "Which email body goes with this subject?",
    "Which email was sent under this subject line?",
    "Select the email this subject belongs to.",
    "Which message does this subject line describe?",
    "Which customer email has this subject?",
]


def ticket_split(body: str) -> str:
    bucket = int.from_bytes(hashlib.sha1(body.encode()).digest()[:8], "big") % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


class TicketSource(JevDataset):
    """The helpdesk tickets, split once by body, so every ticket dataset puts a ticket in the same partition."""

    def load(self) -> DatasetDict:
        rows = load_dataset(self.hf_path)["train"]
        splits = [ticket_split(body or "") for body in rows["body"]]
        return DatasetDict(
            {
                split: rows.select([i for i, s in enumerate(splits) if s == split])
                for split in ("train", "validation", "test")
            }
        )


class TicketDataset(TicketSource):
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


class TicketSubjectDataset(TicketSource):
    """Retrieval: the email body that goes with a subject line, among bodies of other tickets in the same queue."""

    type = DatasetType.CHOICE

    def prepare(self):
        everything = [row for rows in self.data.values() for row in rows]
        bodies = Counter(row["body"] for row in everything)
        subjects = Counter((row["subject"] or "").strip().casefold() for row in everything)

        def usable(row: dict) -> bool:
            # Generic subjects ("Billing Issue") repeat and fit many bodies, so only distinctive ones are queries.
            subject = (row["subject"] or "").strip()
            return (
                bool((row["body"] or "").strip())
                and len(subject.split()) >= 3
                and subjects[subject.casefold()] == 1
                and bodies[row["body"]] == 1
            )

        self.data = DatasetDict({split: self.split(list(filter(usable, rows))) for split, rows in self.data.items()})

    @staticmethod
    def split(rows: list[dict]) -> Dataset:
        # Distractors come from the same split, so held-out emails never appear in training samples.
        by_queue, by_language = defaultdict(list), defaultdict(list)
        for row in rows:
            by_queue[row["language"], row["queue"]].append(row)
            by_language[row["language"]].append(row)
        samples = []
        for row in rows:
            subject = row["subject"].strip()
            rng = random.Random(f"subject:{subject}")
            # One subject in five loses its email, so a match among the options is never a given.
            with_gold = rng.random() >= 0.2
            wanted = 3 if with_gold else 4
            distractors = []
            # Same-queue bodies first: they share the topic, so the subject's specifics have to decide.
            for pool in (by_queue[row["language"], row["queue"]], by_language[row["language"]]):
                for other in rng.sample(pool, min(len(pool), wanted * 3)):
                    if other is not row and other["body"] not in distractors and len(distractors) < wanted:
                        distractors.append(other["body"])
            if len(distractors) < wanted:
                continue
            texts = stable_shuffle(([row["body"]] if with_gold else []) + distractors, subject)
            ids = list("ABCD")
            options = make_options(ids, texts) + make_options(["none"], ["None of these emails"])
            label = ids[texts.index(row["body"])] if with_gold else "none"
            state = with_context(SUBJECT_ABOUT, subject, "Subject")
            question = pick_question(SUBJECT_QUESTIONS, subject)
            samples.append(make_sample(state, question, stable_shuffle(options, f"options:{subject}"), label))
        return Dataset.from_list(samples, features=SAMPLE_FEATURES)


ticket_subject_dataset = TicketSubjectDataset(name="ticket_subject", hf_path=HF_PATH)
