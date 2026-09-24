from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Is this text toxic?"
OPTIONS = make_options(["yes", "no"])


class MultilingualToxicityDataset(JevDataset):
    def prepare(self):
        # splits are languages ("en", "ru", "uk", ...), not train/test
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, OPTIONS, "yes" if x["toxic"] else "no"),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


multilingual_toxicity_dataset = MultilingualToxicityDataset(
    name="multilingual_toxicity", hf_path="textdetox/multilingual_toxicity_dataset"
)
