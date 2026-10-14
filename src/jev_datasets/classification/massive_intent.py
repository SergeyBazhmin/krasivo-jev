from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Which intent does this request to a virtual assistant express?",
    "What does the user want the assistant to do?",
    "What is the intent of this voice assistant command?",
    "Which assistant intent matches this utterance?",
    "How should the assistant interpret this request?",
]
ABOUT = [
    (
        "You are a voice assistant. A user has given the command below. Decide which intent it expresses. Each "
        'intent is named by its scenario and then its action: "alarm set" sets an alarm, "alarm query" asks about '
        'alarms already set, "iot hue lightoff" turns off smart lights, "qa factoid" asks a factual question.'
    ),
    (
        "You are a voice assistant, and a user has given you the command below. Decide which intent it expresses. "
        'An intent\'s name gives its scenario, then its action: "alarm set" sets an alarm, "alarm query" asks '
        'about alarms already set, "iot hue lightoff" turns off smart lights, "qa factoid" asks a factual '
        "question."
    ),
    (
        "Below is a command a user gave to a voice assistant. Classify it by intent. Intent names give a scenario "
        'followed by an action: "alarm set" sets an alarm, "alarm query" asks about alarms already set, "iot hue '
        'lightoff" turns smart lights off, and "qa factoid" asks a factual question.'
    ),
    (
        "As a voice assistant, you map each command to an intent named by scenario and then action. For example "
        '"alarm set" sets an alarm, "alarm query" asks about alarms that are already set, "iot hue lightoff" turns '
        'off smart lights, and "qa factoid" asks a factual question. Decide which intent the command below '
        "expresses."
    ),
    (
        "A user spoke the command below to a voice assistant. Decide which intent it expresses. Each intent is "
        'named scenario first, action second: "alarm set" sets an alarm, "alarm query" asks about alarms already '
        'set, "iot hue lightoff" switches off smart lights, and "qa factoid" asks a factual question.'
    ),
]


class MassiveIntentDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # `label` is the intent name itself, e.g. "alarm_set"
        ids = sorted({id for rows in self.data.values() for id in rows["label"]})
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Command"), pick_question(QUESTIONS, x["text"]), options, x["label"]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


massive_intent_dataset = MassiveIntentDataset(name="massive_intent", hf_path="mteb/amazon_massive_intent", hf_name="en")
