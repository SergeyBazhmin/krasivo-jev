from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample

# every response is rated 0-4 on each attribute; one sample per attribute
ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
OPTIONS = make_options([str(score) for score in range(5)])


def attribute_samples(x: dict) -> list[dict]:
    state = f"Prompt: {x['prompt']}\n\nResponse: {x['response']}"
    return [
        make_sample(
            state,
            f"Rate the {attribute} of the response from 0 (lowest) to 4 (highest).",
            OPTIONS,
            str(x[attribute]),
        )
        for attribute in ATTRIBUTES
    ]


class HelpSteer2Dataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            explode(attribute_samples),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


helpsteer2_dataset = HelpSteer2Dataset(name="helpsteer2", hf_path="nvidia/HelpSteer2")
