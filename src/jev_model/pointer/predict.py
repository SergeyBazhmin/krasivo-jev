from pathlib import Path

import torch

from jev_model.pointer.model import load


class PointerPredictor:
    """A trained decoder and pointer head: one forward pass per question, no generation."""

    def __init__(self, checkpoint_dir: Path, device: str = "cuda"):
        self.model = load(checkpoint_dir, device)

    @torch.inference_mode()
    def predict(self, questions: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        """Probabilities over the options of each (state, question, options) in the given order."""
        encoder = self.model.encoder
        prompts = []
        for (_, _, options), tokens in zip(questions, encoder.tokenize(questions)):
            if len(options) < 2:
                raise ValueError(f"need at least 2 options, got {len(options)}")
            if not encoder.fits(tokens):
                raise ValueError(f"question and options alone exceed {encoder.max_length} tokens")
            prompts.append(encoder.assemble(tokens, list(range(len(options)))))
        logits, _ = self.model(prompts)
        probs = torch.softmax(logits / self.model.head.temperature, dim=-1).cpu()
        return [row[: len(options)].tolist() for row, (_, _, options) in zip(probs, questions)]
