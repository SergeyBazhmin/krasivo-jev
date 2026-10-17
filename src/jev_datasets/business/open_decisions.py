import json

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import GROUPED_FEATURES, make_options, make_sample


def render(state: str | dict | list) -> str:
    # 594 states are JSON objects (a policy plus a message, an account record, ...)
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


class OpenDecisionsDataset(JevDataset):
    """JevBench-style decisions written by an LLM and confirmed by a labeler panel, in 23 languages. A benchmark: no
    paraphrased QUESTIONS/ABOUT and no shuffling, so every model sees each item as published (option order is part
    of the benchmark, see the `option_order` family). Every item is self-contained by construction."""

    benchmark = True

    def __init__(self, name: str, hf_path: str, type: DatasetType):
        super().__init__(name, hf_path)
        self.type = type

    def sample(self, x: dict) -> dict:
        question, criteria = x["question"]["instructions"], x["question"]["criteria"]
        if self.type == DatasetType.NOUL:
            # the true/false criteria define the answer, and a noul question carries only its instructions
            question += f"\nYes: {criteria['true']}\nNo: {criteria['false']}"
            options = make_options(["no", "yes"])
        else:
            options = make_options(
                list(criteria), [f"{key.replace('_', ' ')}: {text}" for key, text in criteria.items()]
            )
        # the single expected answer; the soft `target_probs` of calibration items are left out
        sample = make_sample(render(x["state"]), question, options, x["expected"])
        # `group` ties perturbed and paired variants of one item; most items have none
        return sample | {"group": x["group"] or x["id"]}

    def prepare(self):
        # `score` items (ordered levels) fit neither type, so they are left out
        self.data = self.data.filter(lambda x: x["question"]["type"] == self.type.value).map(
            self.sample, remove_columns=self.source_columns, features=GROUPED_FEATURES
        )


open_decisions_choice_dataset = OpenDecisionsDataset(
    name="open_decisions_choice", hf_path="jgeuter/OpenDecisions", type=DatasetType.CHOICE
)
open_decisions_noul_dataset = OpenDecisionsDataset(
    name="open_decisions_noul", hf_path="jgeuter/OpenDecisions", type=DatasetType.NOUL
)
