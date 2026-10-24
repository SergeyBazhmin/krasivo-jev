"""Scores over padded option logits, [n, max_options], shared by every model: slots past a
sample's `num_options` hold no option."""

from itertools import pairwise

import torch

from jev_model.data import is_benchmark


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


def scores(
    logits: torch.Tensor, labels: torch.Tensor, num_options: torch.Tensor, temperature: float = 1.0
) -> dict[str, float]:
    """Accuracy against the argmax of the label, soft NLL, and top-1 ECE.
    `logits` are already masked (-inf)."""
    logits = logits / temperature
    probs = logits.softmax(dim=-1)
    confidence, predicted = probs.max(dim=-1)
    gold = labels.argmax(dim=-1)
    correct = predicted == gold
    return {
        "n": len(labels),
        "accuracy": correct.float().mean().item(),
        "nll": soft_cross_entropy(logits, labels, num_options).mean().item(),
        "ece": expected_calibration_error(confidence, correct),
    }


def macro_of(per_dataset: list[dict[str, float]]) -> dict[str, float]:
    """The unweighted mean over datasets, so small tasks count as much as large ones."""
    keys = ("accuracy", "nll", "ece")
    macro = {k: sum(s[k] for s in per_dataset) / len(per_dataset) for k in keys}
    return macro | {"n": sum(s["n"] for s in per_dataset)}


def with_macro(per_dataset: dict[str, dict[str, float]]) -> dict[str, dict]:
    """Marks each dataset with `benchmark` and adds `macro` over the other datasets and `benchmark_macro` over the
    benchmarks, each when there is any: benchmarks are reported apart, never averaged in with the rest."""
    marked = {name: s | {"benchmark": is_benchmark(name)} for name, s in per_dataset.items()}
    rest = [s for s in marked.values() if not s["benchmark"]]
    benchmarks = [s for s in marked.values() if s["benchmark"]]
    macros = {"macro": macro_of(rest)} if rest else {}
    if benchmarks:
        macros["benchmark_macro"] = macro_of(benchmarks)
    return marked | macros


def table(metrics: dict[str, dict]) -> str:
    """One line per dataset, as printed by `eval` and at the end of training; benchmarks are marked."""
    return "\n".join(
        f"{name:32} n={m['n']:>6}  acc={m['accuracy']:.3f}  nll={m['nll']:.3f}  ece={m['ece']:.3f}"
        + ("  [benchmark]" if m.get("benchmark") else "")
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


def threshold_scores(p_yes: torch.Tensor, gold_yes: torch.Tensor, thresholds: list[float]) -> list[dict]:
    """Precision, recall and F1 of the positive class when it is predicted at `p_yes >= threshold`, one row per
    threshold. Precision is None when nothing is predicted positive, recall when there is no positive row."""
    rows = []
    for t in thresholds:
        predicted = p_yes >= t
        tp = (predicted & gold_yes).sum().item()
        fp = (predicted & ~gold_yes).sum().item()
        fn = (~predicted & gold_yes).sum().item()
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        rows.append(
            {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": len(gold_yes) - tp - fp - fn,
                "positive_rate": predicted.float().mean().item(),
            }
        )
    return rows


def average_precision(p_yes: torch.Tensor, gold_yes: torch.Tensor) -> float | None:
    """The area under the precision-recall curve, over every cut-off: the mean precision at each positive row's
    rank, ties ranked together."""
    if not gold_yes.any():
        return None
    order = p_yes.argsort(descending=True)
    p, gold = p_yes[order], gold_yes[order].float()
    # the last row of each run of equal probabilities, where a cut-off can fall
    last = torch.cat([p[1:] != p[:-1], torch.tensor([True])])
    tp = gold.cumsum(0)[last]
    precision = tp / (torch.arange(1, len(p) + 1)[last])
    recall_gain = torch.diff(tp, prepend=torch.zeros(1)) / gold.sum()
    return (precision * recall_gain).sum().item()
