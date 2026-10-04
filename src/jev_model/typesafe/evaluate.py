import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from jev_model.data import load_samples
from jev_model.metrics import scores, with_macro
from jev_model.typesafe.client import Client, kind
from jev_model.typesafe.config import TypesafeConfig


def texts(sample: dict) -> list[str]:
    return [option["text"] for option in sample["options"]]


def evaluate(config: TypesafeConfig, names: list[str], part: str, out_dir: Path):
    """Per-dataset scores of the API's probabilities, plus `macro`. The answers are kept in
    OUT_DIR/<part>-<dataset>.jsonl. Two progress bars: every sample of the run, and the current dataset with its
    accuracy so far; each finished dataset prints its scores."""
    client = Client(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_samples(names, config.data_dir, part, config.max_samples)
    metrics = {}
    total = tqdm(total=sum(map(len, data.values())), desc=f"{part} ({len(data)} datasets)", unit="sample")
    for k, (name, samples) in enumerate(data.items(), start=1):
        gold = [max(range(len(s["label"])), key=lambda j: s["label"][j]) for s in samples]
        counts = {"done": 0, "correct": 0}
        current = tqdm(total=len(samples), desc=f"[{k}/{len(data)}] {name}", unit="sample", leave=False)

        def on_answer(i: int, probs: list[float] | None):
            counts["done"] += 1
            counts["correct"] += probs is not None and max(range(len(probs)), key=probs.__getitem__) == gold[i]
            current.set_postfix(acc=f"{counts['correct'] / counts['done']:.3f}", refresh=False)
            current.update()
            total.update()

        questions = [(s["state"], s["question"], texts(s), s.get("type") or kind(texts(s))) for s in samples]
        answers = client.predict(questions, on_answer)
        current.close()
        metrics[name] = score(samples, answers, out_dir / f"{part}-{name}.jsonl")
        m = metrics[name]
        tqdm.write(
            f"[{k}/{len(data)}] {name:32} n={m['n']:>6}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
            f"nll={m['nll']:.3f}  ece={m['ece']:.3f}  answered={m['answered']:.3f}"
        )
        total.set_postfix(macro_acc=f"{sum(m['accuracy'] for m in metrics.values()) / len(metrics):.3f}")
    total.close()
    return with_macro(metrics)


def score(samples: list[dict], answers: list[list[float] | None], path: Path) -> dict[str, float]:
    """`jev_model.metrics.scores` over the log probabilities. A rejected sample counts as uniform, which scores as
    the first option. Probabilities are floored so a confident miss has a finite NLL."""
    width = max(len(s["options"]) for s in samples)
    logits = torch.full((len(samples), width), -math.inf)
    labels = torch.zeros(len(samples), width)
    num_options = torch.tensor([len(s["options"]) for s in samples])
    with path.open("w") as f:
        for i, (sample, probs) in enumerate(zip(samples, answers)):
            n = len(sample["options"])
            p = torch.tensor(probs) if probs else torch.full((n,), 1 / n)
            logits[i, :n] = p.clamp(min=1e-9).log()
            labels[i, :n] = torch.tensor(sample["label"])
            f.write(
                json.dumps(
                    {"question": sample["question"], "options": texts(sample), "probs": probs, "label": sample["label"]}
                )
                + "\n"
            )
    return scores(logits, labels, num_options) | {"answered": sum(a is not None for a in answers) / len(answers)}
