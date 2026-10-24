"""Named yes/no and choice questions about one state, answered by a trained model offline, in the shape of TypeSafe's
`system_one`. The model loads from a run, a stage directory or a Hugging Face repo holding a stage directory."""

from dataclasses import dataclass
from pathlib import Path

from jev_model.base import RUN_FILE, Predictor

# a noul's options, in the order the model sees them, whatever the language of the instructions
YES_NO = ["yes", "no"]


@dataclass
class Noul:
    instructions: str


@dataclass
class Choice:
    instructions: str
    # label -> when it applies; None leaves the label to speak for itself
    criteria: dict[str, str | None]


@dataclass
class NoulAnswer:
    # probability of yes
    noul: float


@dataclass
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


def options(question: Noul | Choice) -> list[str]:
    if isinstance(question, Noul):
        return list(YES_NO)
    if isinstance(question, Choice):
        if len(question.criteria) < 2:
            raise ValueError(f"a choice needs at least 2 criteria, got {len(question.criteria)}")
        return [f"{key}: {text}" if text else key for key, text in question.criteria.items()]
    raise ValueError(f"unsupported question type {type(question).__name__}; use Noul or Choice")


def answer(question: Noul | Choice, probs: list[float]) -> NoulAnswer | ChoiceAnswer:
    if isinstance(question, Noul):
        return NoulAnswer(noul=probs[0])
    probabilities = dict(zip(question.criteria, probs))
    choice = max(probabilities, key=probabilities.__getitem__)
    return ChoiceAnswer(choice=choice, confidence=probabilities[choice], probabilities=probabilities)


class JevHolder:
    """A loaded model that answers `system_one` questions. Every question of a call goes in one batch."""

    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    @classmethod
    def from_pretrained(
        cls, source: str | Path, device: str = "", stage: str | None = None, revision: str | None = None
    ) -> "JevHolder":
        """`source` is a run directory, a stage directory (`adapter/` and `head.pt`), an export folder, or a Hugging
        Face repo id pushed from one (`jev-model push`). The base model is fetched by the name the checkpoint stores."""
        from jev_model.cli import default_device, open_run

        device = default_device(device)
        path = Path(source)
        if (path / RUN_FILE).is_file():
            model, config = open_run(path)
            return cls(model.predictor(config, path, device, stage))
        from jev_model.pointer.model import resolve_checkpoint
        from jev_model.pointer.predict import PointerPredictor

        return cls(PointerPredictor(resolve_checkpoint(source, revision), device))

    def system_one(self, state: str, questions: dict[str, Noul | Choice]) -> dict[str, NoulAnswer | ChoiceAnswer]:
        """An answer per question name."""
        if not questions:
            raise ValueError("questions must not be empty")
        batch = []
        for question in questions.values():
            # options first: they reject an unsupported question type before its fields are read
            texts = options(question)
            batch.append((state, question.instructions, texts))
        probs = self.predictor.predict(batch)
        return {name: answer(q, p) for (name, q), p in zip(questions.items(), probs)}
