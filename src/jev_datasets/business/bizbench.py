from string import ascii_uppercase

from datasets import DatasetDict

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class BizBenchFinKnowDataset(JevDataset):
    """BizBench's FinKnow task: CFA-style finance questions; the other tasks are open-ended."""

    def prepare(self):
        self.data = self.data.filter(lambda x: x["task"] == "FinKnow")
        # FinKnow only ships a test split
        self.data = DatasetDict({split: rows for split, rows in self.data.items() if rows.num_rows})
        # `answer` is the index of the correct option, as a string
        self.data = self.data.map(
            lambda x: make_sample(
                "",
                x["question"],
                make_options(list(ascii_uppercase[: len(x["options"])]), x["options"]),
                ascii_uppercase[int(x["answer"])],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


bizbench_finknow_dataset = BizBenchFinKnowDataset(name="bizbench_finknow", hf_path="kensho/bizbench")
