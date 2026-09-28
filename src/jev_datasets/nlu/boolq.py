from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

OPTIONS = make_options(["yes", "no"])
ABOUT = (
    "You are given a passage from an encyclopedia article and a yes/no question someone typed into a search engine, "
    "lowercased and without a question mark. Answer the question from the passage."
)


class BoolQDataset(JevDataset):
    type = DatasetType.NOUL

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["passage"], "Passage"),
                x["question"],
                OPTIONS,
                "yes" if x["answer"] else "no",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


boolq_dataset = BoolQDataset(name="boolq", hf_path="google/boolq")
