from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "What is this airline customer asking about?"

# Atomic intents seen in train; rows may join several with "+" (e.g. "flight+airfare").
# "day_name" only occurs in test and is dropped.
INTENTS = [
    "flight",
    "airfare",
    "ground_service",
    "airline",
    "abbreviation",
    "aircraft",
    "flight_time",
    "quantity",
    "airport",
    "distance",
    "city",
    "ground_fare",
    "capacity",
    "flight_no",
    "meal",
    "restriction",
    "cheapest",
]
OPTIONS = make_options(INTENTS, [intent.replace("_", " ") for intent in INTENTS])


def intent_distribution(intent: str) -> list[float]:
    """Joint intents share the probability mass evenly."""
    parts = intent.split("+")
    return [parts.count(id) / len(parts) for id in INTENTS]


class ATISDataset(JevDataset):
    def prepare(self):
        self.data = self.data.filter(lambda x: all(part in INTENTS for part in x["intent"].split("+")))
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, OPTIONS, intent_distribution(x["intent"])),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


atis_dataset = ATISDataset(name="atis", hf_path="tuetschek/atis")
