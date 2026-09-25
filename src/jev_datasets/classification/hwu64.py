from datasets import load_dataset

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which intent does this request to a virtual assistant express?"


class HWU64Dataset(JevDataset):
    def prepare(self):
        # `label` is a bare int; its names ("alarm_query", ...) live in the separate `intents` config
        intents = load_dataset(self.hf_path, "intents", split="intents")
        ids = [name for _, name in sorted(zip(intents["id"], intents["name"]))]
        options = make_options(ids, [id.replace("_", " ") for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(x["utterance"], QUESTION, options, ids[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


hwu64_dataset = HWU64Dataset(name="hwu64", hf_path="DeepPavlov/hwu64")
