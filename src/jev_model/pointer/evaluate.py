import json
from pathlib import Path

import torch
from tqdm import tqdm

from jev_model import distributed
from jev_model.metrics import fit_temperature, scores, with_macro
from jev_model.pointer.data import Item, token_batches
from jev_model.pointer.model import PointerModel

# (logits, labels, num_options) of one dataset, padded to its widest sample
Outputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@torch.inference_mode()
def outputs_of(model: PointerModel, items: list[Item], batch_tokens: int, desc: str = "") -> Outputs:
    """Uncalibrated logits with the options in the source order."""
    model.eval()
    num_options = torch.tensor([len(item.label) for item in items])
    slots = int(num_options.max())
    logits = torch.full((len(items), slots), float("-inf"))
    labels = torch.zeros((len(items), slots))
    for i, item in enumerate(items):
        labels[i, : len(item.label)] = torch.tensor(item.label)
    batches = list(token_batches(list(range(len(items))), [item.length for item in items], batch_tokens))
    for batch in tqdm(batches, desc=desc, leave=False, disable=not distributed.is_main()):
        prompts = [model.encoder.assemble(items[i].tokens, list(range(len(items[i].label)))) for i in batch]
        out = model(prompts)[0].cpu()
        logits[torch.tensor(batch), : out.shape[1]] = out
    return logits, labels, num_options


def run(model: PointerModel, data: dict[str, list[Item]], batch_tokens: int) -> dict[str, Outputs]:
    return {name: outputs_of(model, items, batch_tokens, desc=name) for name, items in data.items()}


def score(outputs: dict[str, Outputs], temperature: float) -> dict[str, dict[str, float]]:
    """Per-dataset scores, plus the macros of `with_macro`."""
    return with_macro({name: scores(*out, temperature) for name, out in outputs.items()})


def evaluate(
    model: PointerModel, data: dict[str, list[Item]], batch_tokens: int, out_dir: Path | None = None, part: str = ""
) -> dict[str, dict[str, float]]:
    """Per-dataset scores with the head's fitted temperature, plus the macros of `with_macro`. With `out_dir`, the
    calibrated probabilities are kept in OUT_DIR/<part>-<dataset>.jsonl, as `typesafe` keeps its answers."""
    outputs = run(model, data, batch_tokens)
    temperature = model.head.temperature.item()
    if out_dir is not None and distributed.is_main():
        for name, (logits, _, _) in outputs.items():
            save_predictions(data[name], (logits / temperature).softmax(dim=-1), out_dir / f"{part}-{name}.jsonl")
    return score(outputs, temperature)


def save_predictions(items: list[Item], probs: torch.Tensor, path: Path):
    """One line per item: its options, the probabilities over them and the label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item, p in zip(items, probs):
            n = len(item.label)
            f.write(json.dumps({"options": item.options, "probs": p[:n].tolist(), "label": item.label}) + "\n")


def pooled_temperature(outputs: dict[str, Outputs]) -> float:
    """One temperature for every dataset, fitted on the pooled rows."""
    slots = max(logits.shape[1] for logits, _, _ in outputs.values())

    def widen(tensor: torch.Tensor, fill: float) -> torch.Tensor:
        return torch.nn.functional.pad(tensor, (0, slots - tensor.shape[1]), value=fill)

    return fit_temperature(
        torch.cat([widen(logits, float("-inf")) for logits, _, _ in outputs.values()]),
        torch.cat([widen(labels, 0.0) for _, labels, _ in outputs.values()]),
        torch.cat([num_options for _, _, num_options in outputs.values()]),
    )
