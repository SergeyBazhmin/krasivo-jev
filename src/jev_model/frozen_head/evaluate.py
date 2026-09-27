import torch

from jev_model.frozen_head.head import OptionHead
from jev_model.metrics import fit_temperature, scores, with_macro


@torch.no_grad()
def logits_of(head: OptionHead, tensors: dict[str, torch.Tensor], device: str, batch_size: int = 4096) -> torch.Tensor:
    head.eval()
    out = [
        head(tensors["features"][i : i + batch_size].to(device), tensors["num_options"][i : i + batch_size].to(device))
        for i in range(0, len(tensors["features"]), batch_size)
    ]
    return torch.cat(out).cpu()


def evaluate(head: OptionHead, data: dict[str, dict[str, torch.Tensor]], device: str) -> dict[str, dict[str, float]]:
    """Per-dataset scores with the head's fitted temperature, plus `macro`."""
    return with_macro({
        name: scores(logits_of(head, tensors, device), tensors["labels"], tensors["num_options"], head.temperature.item())
        for name, tensors in data.items()
    })


def calibrate(head: OptionHead, data: dict[str, dict[str, torch.Tensor]], device: str) -> float:
    """Fits the head's temperature on `data` (validation) and returns it."""
    logits = torch.cat([logits_of(head, tensors, device) for tensors in data.values()])
    labels = torch.cat([tensors["labels"] for tensors in data.values()])
    num_options = torch.cat([tensors["num_options"] for tensors in data.values()])
    head.temperature.fill_(fit_temperature(logits, labels, num_options))
    return head.temperature.item()
