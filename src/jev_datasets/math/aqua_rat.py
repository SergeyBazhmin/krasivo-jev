from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_option, make_sample


def parse_option(option: str) -> dict[str, str]:
    """ "A)21" -> {"id": "A", "text": "21"}"""
    id, text = option.split(")", 1)
    return make_option(id.strip(), text.strip())


class AQuARatDataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                "",
                x["question"],
                [parse_option(option) for option in x["options"]],
                x["correct"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


aqua_rat_dataset = AQuARatDataset(name="aqua_rat", hf_path="deepmind/aqua_rat", hf_name="raw")
