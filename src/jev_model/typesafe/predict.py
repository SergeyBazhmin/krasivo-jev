from jev_model.typesafe.client import Client
from jev_model.typesafe.config import TypesafeConfig


class TypesafePredictor:
    """The API's probabilities over the options; uniform for a sample it rejects."""

    def __init__(self, config: TypesafeConfig):
        self.client = Client(config)

    def predict(self, questions: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        for _, _, options in questions:
            if len(options) < 2:
                raise ValueError(f"need at least 2 options, got {len(options)}")
        return [
            probs or [1 / len(options)] * len(options)
            for (_, _, options), probs in zip(questions, self.client.predict(questions))
        ]
