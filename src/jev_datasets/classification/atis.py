from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is this airline customer asking about?",
    "What information is this traveler looking for?",
    "What is the intent of this flight booking query?",
    "Which kind of travel information does this request ask for?",
    "What does this passenger want to know?",
]

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
ABOUT = (
    "You are an automated flight information system. A traveler has spoken the request below. Decide what information "
    "they are asking for; a request that asks for two things at once counts toward both."
)
ANSWERS = {
    "flight": "flights between places, on given dates or times",
    "airfare": "the price of a flight",
    "ground service": "transport on the ground, such as taxis, rental cars or limousines",
    "airline": "which airlines fly somewhere, or facts about an airline",
    "abbreviation": "what a code or abbreviation means, such as a fare class or airline code",
    "aircraft": "the type of plane used",
    "flight time": "departure or arrival times of flights",
    "quantity": "how many of something there are, such as flights or airlines",
    "airport": "airports in or near a place",
    "distance": "how far apart two places are, such as an airport and a city",
    "city": "which city an airport or airline is in or serves",
    "ground fare": "the price of ground transport",
    "capacity": "how many passengers a plane can seat",
    "flight no": "the flight number",
    "meal": "meals served on a flight",
    "restriction": "the rules and restrictions attached to a fare",
    "cheapest": "the cheapest option",
}
OPTIONS = make_options(INTENTS, [intent.replace("_", " ") for intent in INTENTS])


def intent_distribution(intent: str) -> list[float]:
    """Joint intents share the probability mass evenly."""
    parts = intent.split("+")
    return [parts.count(id) / len(parts) for id in INTENTS]


class ATISDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.filter(lambda x: all(part in INTENTS for part in x["intent"].split("+")))
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Request", ANSWERS),
                pick_question(QUESTIONS, x["text"]),
                OPTIONS,
                intent_distribution(x["intent"]),
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


atis_dataset = ATISDataset(name="atis", hf_path="tuetschek/atis")
