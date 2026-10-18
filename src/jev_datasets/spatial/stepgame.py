from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

# the relation of the first agent to the second on a grid
RELATIONS = ["left", "right", "above", "below", "upper-left", "upper-right", "lower-left", "lower-right", "overlap"]
OPTIONS = make_options(RELATIONS, [relation.replace("-", " ") for relation in RELATIONS])

ABOUT = [
    (
        "You are given the positions of agents on a grid. Each sentence says where one agent is relative to "
        "another, in words or as a clock position (12 o'clock is above, 3 o'clock right, 6 o'clock below, 9 "
        "o'clock left, and the hours in between diagonal). Chain the steps to find where the first agent asked "
        'about is relative to the second; "overlap" means they are at the same spot.'
    ),
    (
        "Below are the positions of agents on a grid. Each sentence places one agent relative to another, in words "
        "or as a clock position (12 o'clock is above, 3 o'clock right, 6 o'clock below, 9 o'clock left, and the "
        "hours in between diagonal). Chain the steps to find where the first agent asked about is relative to the "
        'second; "overlap" means they are at the same spot.'
    ),
    (
        "You are given agents placed on a grid. Every sentence tells where one agent is relative to another, "
        "either in words or as a clock position: 12 o'clock above, 3 o'clock right, 6 o'clock below, 9 o'clock "
        "left, and the hours in between diagonal. Follow the steps to find where the first agent asked about is "
        'relative to the second; "overlap" means the same spot.'
    ),
    (
        "You track agents on a grid. Each sentence gives one agent's position relative to another, in words or as "
        "a clock position, where 12 o'clock is above, 3 o'clock right, 6 o'clock below, 9 o'clock left, and the "
        "hours in between are diagonal. Chain these steps to locate the first agent asked about relative to the "
        'second; "overlap" means they share a spot.'
    ),
    (
        "Agents are placed on a grid. Sentences describe one agent's position relative to another, in words or as "
        "a clock position (12 o'clock above, 3 o'clock right, 6 o'clock below, 9 o'clock left, hours in between "
        "diagonal). Chain the steps to find where the first agent asked about is relative to the second, where "
        '"overlap" means the same spot.'
    ),
]


class StepGameDataset(JevDataset):
    type = DatasetType.CHOICE
    # its topic's benchmark pick: scored by every model, trained on only with `train_on_benchmark`
    benchmark = True

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(with_context(ABOUT, x["story"], "Story"), x["question"], OPTIONS, x["label"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


stepgame_dataset = StepGameDataset(name="stepgame", hf_path="tasksource/stepgame")
