from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class JevDistillDataset(JevDataset):
    """Distillation corpus that already ships the contract: soft `target` over plain-string options."""

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                x["state"],
                x["question"],
                make_options(x["options"]),
                x["target"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


jev_distill_dataset = JevDistillDataset(
    name="jev_distill", hf_path="SargeDev/jev-distill-corpus-v3"
)
