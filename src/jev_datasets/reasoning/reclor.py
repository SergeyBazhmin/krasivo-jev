from string import ascii_uppercase

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = (
    "You are taking a logical reasoning test. Read the passage and answer the question about its argument or "
    "situation, for example what it assumes, what would strengthen or weaken it, what follows from it, or where its "
    "reasoning goes wrong."
)


class ReClorDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["context"], "Passage"),
                x["question"],
                make_options(list(ascii_uppercase[: len(x["answers"])]), x["answers"]),
                ascii_uppercase[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


reclor_dataset = ReClorDataset(name="reclor", hf_path="tasksource/reclor")
