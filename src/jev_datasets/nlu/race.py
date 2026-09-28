from string import ascii_uppercase

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = (
    "You are taking an English reading comprehension exam. Read the passage and answer the question about it; a "
    "question with a blank (_) asks which option completes it."
)


class RACEDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["article"], "Passage"),
                x["question"],
                make_options(list(ascii_uppercase[: len(x["options"])]), x["options"]),
                x["answer"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


race_dataset = RACEDataset(name="race", hf_path="ehovy/race", hf_name="all")
