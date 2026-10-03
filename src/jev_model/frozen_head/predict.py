from pathlib import Path

import torch

from jev_model.frozen_head.backbone import embed, load_backbone
from jev_model.frozen_head.head import OptionHead
from jev_model.prompt import encode


class JevPredictor:
    """Frozen backbone and a trained head: one forward pass per question, no generation."""

    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.cache = checkpoint["cache"]
        self.head = OptionHead(**checkpoint["head_config"])
        self.head.load_state_dict(checkpoint["head"])
        self.head.to(device).eval()
        self.decoder, self.tokenizer = load_backbone(self.cache["model"], device)
        self.pad_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        self.device = device

    def predict(self, questions: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        """Probabilities over the options of each (state, question, options) in the given order."""
        prompts = []
        for state, question, options in questions:
            if not 2 <= len(options) <= self.head.max_options:
                raise ValueError(f"need 2..{self.head.max_options} options, got {len(options)}")
            ids = encode(self.tokenizer, state, question, options, self.cache["max_length"])
            if ids is None:
                raise ValueError(f"question and options alone exceed {self.cache['max_length']} tokens")
            prompts.append(ids)
        hidden = embed(self.decoder, self.pad_id, prompts, self.cache["layer"]).to(self.device)
        num_options = torch.tensor([len(options) for _, _, options in questions], device=self.device)
        probs = self.head.probabilities(hidden, num_options).cpu()
        return [row[: len(options)].tolist() for row, (_, _, options) in zip(probs, questions)]
