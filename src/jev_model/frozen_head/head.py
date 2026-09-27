import torch
from torch import nn

from jev_model.frozen_head.config import MAX_OPTIONS
from jev_model.metrics import option_mask


class OptionHead(nn.Module):
    """Last-token hidden state -> one logit per option slot. Slots past a sample's option count
    are masked, so the softmax runs over exactly the options that were in the prompt."""

    def __init__(self, hidden_size: int, width: int = 2048, depth: int = 2, dropout: float = 0.1,
                 max_options: int = MAX_OPTIONS):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(hidden_size)]
        size = hidden_size
        for _ in range(depth - 1):
            layers += [nn.Linear(size, width), nn.GELU(), nn.Dropout(dropout)]
            size = width
        layers.append(nn.Linear(size, max_options))
        self.net = nn.Sequential(*layers)
        self.max_options = max_options
        # fitted on validation after training; divides the logits at inference
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, hidden: torch.Tensor, num_options: torch.Tensor) -> torch.Tensor:
        logits = self.net(hidden.float())
        return logits.masked_fill(option_mask(num_options, self.max_options), float("-inf"))

    @torch.no_grad()
    def probabilities(self, hidden: torch.Tensor, num_options: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(hidden, num_options) / self.temperature, dim=-1)

