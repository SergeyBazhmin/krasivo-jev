from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class JevDistillDataset(JevDataset):
    """Distillation corpus that already ships the contract: soft `target` over plain-string options."""

    def prepare(self):
        # Only `openjev_v2` (games, geometry, workflow traces) is answerable from the state alone:
        # `yuri_v3` asks for expert judgement (process hazard, pathogenicity, ...) and `yuri_v1`
        # memory relevance is all uniform [0.5, 0.5]. Uniform targets carry no answer either way.
        self.data = self.data.filter(lambda x: x["source"] == "openjev_v2" and len(set(x["target"])) > 1)
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
