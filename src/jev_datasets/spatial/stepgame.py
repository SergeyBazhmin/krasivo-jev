from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

# the relation of the first agent to the second on a grid
RELATIONS = ["left", "right", "above", "below", "upper-left", "upper-right", "lower-left", "lower-right", "overlap"]
OPTIONS = make_options(RELATIONS, [relation.replace("-", " ") for relation in RELATIONS])


class StepGameDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(x["story"], x["question"], OPTIONS, x["label"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


stepgame_dataset = StepGameDataset(name="stepgame", hf_path="tasksource/stepgame")
