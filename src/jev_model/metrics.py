"""Scores over padded option logits, [n, max_options], shared by every model: slots past a
sample's `num_options` hold no option."""

from itertools import pairwise

import torch


def option_mask(num_options: torch.Tensor, max_options: int) -> torch.Tensor:
    """True at the slots that hold no option."""
    return torch.arange(max_options, device=num_options.device) >= num_options.long().unsqueeze(-1)


def masked_log_probs(logits: torch.Tensor, num_options: torch.Tensor) -> torch.Tensor:
    """log_softmax with 0 (not -inf) at the empty slots, so `labels * log_probs` stays finite."""
    return logits.log_softmax(dim=-1).masked_fill(option_mask(num_options, logits.shape[-1]), 0.0)


def soft_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, num_options: torch.Tensor) -> torch.Tensor:
    return -(labels * masked_log_probs(logits, num_options)).sum(dim=-1)


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        inside = (confidence > lo) & (confidence <= hi)
        if inside.any():
            ece += inside.float().mean().item() * abs(
                confidence[inside].mean().item() - correct[inside].float().mean().item()
            )
    return ece


def macro_prf(predicted: torch.Tensor, gold: torch.Tensor) -> dict[str, float]:
    """Precision, recall and F1 per option slot, averaged over slots that occur as gold or prediction.
    Classes are slots, not option ids: that matches fixed-option datasets, and on shuffled ones shows positional bias."""
    classes = torch.cat([predicted, gold]).unique()
    hit = (predicted.unsqueeze(-1) == classes) & (gold.unsqueeze(-1) == classes)
    tp = hit.sum(dim=0).float()
    precision = tp / (predicted.unsqueeze(-1) == classes).sum(dim=0).clamp(min=1)
    recall = tp / (gold.unsqueeze(-1) == classes).sum(dim=0).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {"precision": precision.mean().item(), "recall": recall.mean().item(), "f1": f1.mean().item()}


def scores(
    logits: torch.Tensor, labels: torch.Tensor, num_options: torch.Tensor, temperature: float = 1.0
) -> dict[str, float]:
    """Accuracy and macro precision/recall/F1 against the argmax of the label, soft NLL, and top-1 ECE.
    `logits` are already masked (-inf)."""
    logits = logits / temperature
    probs = logits.softmax(dim=-1)
    confidence, predicted = probs.max(dim=-1)
    gold = labels.argmax(dim=-1)
    correct = predicted == gold
    return {
        "n": len(labels),
        "accuracy": correct.float().mean().item(),
        **macro_prf(predicted, gold),
        "nll": soft_cross_entropy(logits, labels, num_options).mean().item(),
        "ece": expected_calibration_error(confidence, correct),
    }


def with_macro(per_dataset: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Adds `macro`, the unweighted mean over datasets, so small tasks count as much as large ones."""
    keys = ("accuracy", "precision", "recall", "f1", "nll", "ece")
    macro = {k: sum(s[k] for s in per_dataset.values()) / len(per_dataset) for k in keys}
    return per_dataset | {"macro": macro | {"n": sum(s["n"] for s in per_dataset.values())}}


def table(metrics: dict[str, dict[str, float]]) -> str:
    """One line per dataset, as printed by `eval` and at the end of training."""
    return "\n".join(
        f"{name:32} n={m['n']:>6}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  nll={m['nll']:.3f}  ece={m['ece']:.3f}"
        for name, m in metrics.items()
    )


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, num_options: torch.Tensor) -> float:
    """One temperature minimising the pooled NLL, so larger datasets weigh more, as in training."""
    # finite while scaled: d(-inf / t)/dt would turn the gradient into NaN
    mask = option_mask(num_options, logits.shape[-1])
    logits = logits.masked_fill(mask, 0.0)
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        scaled = (logits / log_t.exp()).masked_fill(mask, float("-inf"))
        loss = soft_cross_entropy(scaled, labels, num_options).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()
