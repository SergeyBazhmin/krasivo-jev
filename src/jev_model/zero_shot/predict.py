from jev_model.zero_shot.client import connect
from jev_model.zero_shot.config import ZeroShotConfig
from jev_model.zero_shot.prompt import parse


class ZeroShotPredictor:
    """The served LM as it is: one-hot on the option its greedy answer names, uniform when it names none."""

    def __init__(self, config: ZeroShotConfig):
        self.config = config
        self.client, self.prompter = connect(config)

    def predict(self, questions: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        """One-hot over the options of each (state, question, options), in the given order."""
        messages = []
        for state, question, options in questions:
            if len(options) < 2:
                raise ValueError(f"need at least 2 options, got {len(options)}")
            if (message := self.prompter(state, question, options)) is None:
                raise ValueError(f"question and options alone exceed {self.config.max_length} tokens")
            messages.append(message)
        result = []
        for (_, _, options), answer in zip(questions, self.client.answer(messages)):
            choice = parse(answer, options)
            result.append(
                [1 / len(options)] * len(options)
                if choice is None
                else [float(i == choice) for i in range(len(options))]
            )
        return result
