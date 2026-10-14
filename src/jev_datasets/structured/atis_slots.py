from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, pick_question, stable_shuffle, text_choice

QUESTIONS = [
    "Which phrase in the request gives {slot}?",
    "What does the traveler say for {slot}?",
    "Choose the words of the request that name {slot}.",
    "Which part of the request specifies {slot}?",
    "Identify {slot} in the traveler's own words.",
]
ABOUT = (
    "You are an automated flight information system. A traveler has spoken the request below. Find the words that "
    "fill the requested field, using only what the traveler said."
)
# Concrete spans whose role is stated in the request. Bare modifiers ("cheapest", "before", "round trip"),
# undirected cities and codes such as fare bases are left out: their meaning comes from the ATIS ontology.
SLOTS = {
    "fromloc.city_name": "the departure city",
    "toloc.city_name": "the destination city",
    "stoploc.city_name": "the stopover city",
    "fromloc.airport_name": "the departure airport",
    "toloc.airport_name": "the destination airport",
    "fromloc.state_name": "the departure state",
    "toloc.state_name": "the destination state",
    "depart_date.day_name": "the day of the week of departure",
    "depart_date.day_number": "the day of the month of departure",
    "depart_date.month_name": "the month of departure",
    "depart_date.today_relative": "the departure day relative to today",
    "depart_time.time": "the departure time",
    "depart_time.period_of_day": "the part of the day for departure",
    "depart_time.start_time": "the earliest departure time",
    "depart_time.end_time": "the latest departure time",
    "arrive_date.day_name": "the day of the week of arrival",
    "arrive_date.day_number": "the day of the month of arrival",
    "arrive_date.month_name": "the month of arrival",
    "arrive_time.time": "the arrival time",
    "arrive_time.period_of_day": "the part of the day for arrival",
    "arrive_time.start_time": "the earliest arrival time",
    "arrive_time.end_time": "the latest arrival time",
    "airline_name": "the airline",
    "class_type": "the cabin class",
    "flight_number": "the flight number",
    "fare_amount": "the fare amount",
    "meal_description": "the meal",
    "transport_type": "the type of ground transport",
}


def spans(text: str, tags: str) -> list[tuple[str, str]]:
    """(slot, words) for each BIO span; an I- tag that does not continue its span is dropped."""
    result, current = [], None
    for word, tag in zip(text.split(), tags.split()):
        if tag.startswith("B-"):
            current = [tag[2:], [word]]
            result.append(current)
        elif tag.startswith("I-") and current and current[0] == tag[2:]:
            current[1].append(word)
        else:
            current = None
    return [(slot, " ".join(words)) for slot, words in result]


def slot_samples(row: dict) -> list[dict]:
    if len(row["text"].split()) != len(row["slots"].split()):
        return []
    annotated = spans(row["text"], row["slots"])
    samples = []
    for slot in stable_shuffle(sorted({slot for slot, _ in annotated} & SLOTS.keys()), row["text"])[:3]:
        values = {value for key, value in annotated if key == slot}
        # "to denver or dallas" fills one field twice, so neither value alone is the answer.
        if len(values) != 1:
            continue
        answer = values.pop()
        # Every candidate is said in the request: options taken from other requests would be ruled out by
        # string matching alone, while these need the role (departure vs destination, depart vs arrive).
        distractors = [value for key, value in annotated if key != slot and value != answer]
        seed = row["text"] + slot
        question = pick_question(QUESTIONS, seed).format(slot=SLOTS[slot])
        if sample := text_choice(f"{ABOUT}\n\nRequest:\n{row['text']}", question, answer, distractors, seed):
            samples.append(sample)
    return samples


class ATISSlotsDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            explode(slot_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


atis_slots_dataset = ATISSlotsDataset(name="atis_slots", hf_path="tuetschek/atis")
