import math

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which response best follows the instruction?"


def rated(completions: list[dict]) -> list[dict]:
    return [c for c in completions if c["fine-grained_score"] is not None and not math.isnan(c["fine-grained_score"])]


def best_share(scores: list[float]) -> list[float]:
    """All mass on the top-scored responses, split evenly between ties."""
    top = max(scores)
    winners = sum(score == top for score in scores)
    return [float(score == top) / winners for score in scores]


class UltraFeedbackDataset(JevDataset):
    def prepare(self):
        self.data = self.data.filter(lambda x: len(rated(x["completions"])) >= 2)
        self.data = self.data.map(self.to_sample, remove_columns=self.source_columns, features=SAMPLE_FEATURES)

    @staticmethod
    def to_sample(x: dict) -> dict:
        completions = rated(x["completions"])
        return make_sample(
            x["instruction"],
            QUESTION,
            make_options([str(i) for i in range(len(completions))], [c["response"] for c in completions]),
            best_share([c["fine-grained_score"] for c in completions]),
        )


ultrafeedback_dataset = UltraFeedbackDataset(name="ultrafeedback", hf_path="openbmb/UltraFeedback")
