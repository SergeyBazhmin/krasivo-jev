from datasets import DatasetDict

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

IDS = ["A", "B", "C", "D"]


class MERAChoiceDataset(JevDataset):
    """MERA's four-option tasks: `inputs` holds the question and `option_a`..`option_d`."""

    def __init__(self, name: str, task: str):
        super().__init__(name=name, hf_path="MERA-evaluation/MERA", hf_name=task)

    def prepare(self):
        # most tasks hide their test answers (outputs == ""); rummlu publishes them
        self.data = self.data.filter(lambda x: x["outputs"] != "")
        self.data = DatasetDict({split: rows for split, rows in self.data.items() if rows.num_rows})
        self.data = self.data.map(
            lambda x: make_sample(
                x["inputs"].get("subject", ""),
                x["inputs"].get("question") or x["inputs"]["text"],
                make_options(IDS, [x["inputs"][f"option_{id.lower()}"] for id in IDS]),
                x["outputs"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ru_mmlu_dataset = MERAChoiceDataset(name="ru_mmlu", task="rummlu")
ru_openbookqa_dataset = MERAChoiceDataset(name="ru_openbookqa", task="ruopenbookqa")
ru_worldtree_dataset = MERAChoiceDataset(name="ru_worldtree", task="ruworldtree")
