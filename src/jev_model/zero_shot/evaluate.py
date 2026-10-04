import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from jev_model.data import load_samples
from jev_model.metrics import macro_prf, with_macro
from jev_model.zero_shot.client import connect
from jev_model.zero_shot.config import ZeroShotConfig
from jev_model.zero_shot.prompt import parse


def texts(sample: dict) -> list[str]:
    return [option["text"] for option in sample["options"]]


def evaluate(config: ZeroShotConfig, names: list[str], part: str, out_dir: Path):
    """Per-dataset scores of the answers the server generates, plus `macro`. The answers are kept in
    OUT_DIR/<part>-<dataset>.jsonl. Two progress bars: every sample of the run, with its ETA, and the current
    dataset, with its accuracy and parsed share so far; each finished dataset prints its scores."""
    client, prompter = connect(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, samples in load_samples(names, config.data_dir, part, config.max_samples).items():
        built = [(s, prompter(s["state"], s["question"], texts(s))) for s in samples]
        if skipped := sum(p is None for _, p in built):
            tqdm.write(f"[{name}] {skipped} of {len(built)} samples skipped: question and options too long")
        if kept := [(s, p) for s, p in built if p]:
            data[name] = kept
    metrics = {}
    total = tqdm(total=sum(map(len, data.values())), desc=f"{part} ({len(data)} datasets)", unit="sample")
    for k, (name, kept) in enumerate(data.items(), start=1):
        samples, prompts = [s for s, _ in kept], [p for _, p in kept]
        gold = [max(range(len(s["label"])), key=lambda j: s["label"][j]) for s in samples]
        counts = {"done": 0, "parsed": 0, "correct": 0}
        current = tqdm(total=len(prompts), desc=f"[{k}/{len(data)}] {name}", unit="sample", leave=False)

        def on_answer(i: int, answer: str):
            choice = parse(answer, texts(samples[i]))
            counts["done"] += 1
            counts["parsed"] += choice is not None
            counts["correct"] += choice == gold[i]
            current.set_postfix(
                acc=f"{counts['correct'] / counts['done']:.3f}",
                parsed=f"{counts['parsed'] / counts['done']:.2f}",
                refresh=False,
            )
            current.update()
            total.update()

        answers = client.answer(prompts, on_answer)
        current.close()
        metrics[name] = score(samples, answers, out_dir / f"{part}-{name}.jsonl")
        m = metrics[name]
        tqdm.write(
            f"[{k}/{len(data)}] {name:32} n={m['n']:>6}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
            f"parsed={m['parsed']:.3f}"
        )
        total.set_postfix(macro_acc=f"{sum(m['accuracy'] for m in metrics.values()) / len(metrics):.3f}")
    total.close()
    return with_macro(metrics)


def score(samples: list[dict], answers: list[str], path: Path) -> dict[str, float]:
    """Accuracy and macro precision/recall/F1 of the parsed answers; an answer that names no option is wrong.
    `nll` and `ece` need probabilities, which generation does not give: they are NaN."""
    chosen = [parse(answer, texts(sample)) for sample, answer in zip(samples, answers)]
    gold = [max(range(len(s["label"])), key=lambda j: s["label"][j]) for s in samples]
    with path.open("w") as f:
        for sample, answer, choice, g in zip(samples, answers, chosen, gold):
            f.write(
                json.dumps(
                    {
                        "question": sample["question"],
                        "options": texts(sample),
                        "answer": answer,
                        "choice": choice,
                        "gold": g,
                    }
                )
                + "\n"
            )
    predicted = torch.tensor([-1 if choice is None else choice for choice in chosen])
    return {
        "n": len(samples),
        "accuracy": (predicted == torch.tensor(gold)).float().mean().item(),
        **macro_prf(predicted, torch.tensor(gold)),
        "nll": math.nan,
        "ece": math.nan,
        "parsed": sum(choice is not None for choice in chosen) / len(chosen),
    }
