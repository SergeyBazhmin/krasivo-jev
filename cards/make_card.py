"""Plots, metric tables and README.md (from the card's body.md) for a model card of a pointer run.

uv run --with matplotlib python cards/make_card.py pointer-20261017-192041   # cards/<card> for runs/pointer/<run>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from jev_datasets import datasets

REPO = Path(__file__).parent.parent
BASELINE = REPO / "runs/typesafe/20261018-102540/last-test.json"

# reference palette, light mode
POINTER = "#2a78d6"
BASELINE_COLOR = "#eb6834"
BENCHMARK = "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def topic(name: str) -> str:
    return type(datasets[name]).__module__.split(".")[1]


def load_scores(path: Path) -> dict[str, dict]:
    scores = json.loads(path.read_text())
    return {name: s for name, s in scores.items() if name not in ("macro", "benchmark_macro")}


def macro(rows: list[dict], key: str = "accuracy") -> float:
    return sum(r[key] for r in rows) / len(rows)


def training_curve(history: list[dict], plots: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    steps = [h["step"] for h in history]
    for ax, key, title in zip(axes, ("accuracy", "nll", "ece"), ("Accuracy", "NLL", "ECE")):
        ax.plot(steps, [h[key] for h in history], color=POINTER, linewidth=2, marker="o", markersize=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel("optimizer step")
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Validation during the ce stage (macro over datasets, up to 1,000 rows each)",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(plots / "training_curve.png")
    plt.close(fig)


def paired(
    names: list[str],
    ours: dict,
    theirs: dict,
    labels: list[str],
    title: str,
    path: Path,
    row: float,
    xlabel: str = "test accuracy",
):
    """Accuracy of this model and of jev-1.13 side by side, one pair of bars per row, top to bottom."""
    fig, ax = plt.subplots(figsize=(8, row * len(names) + 1.2))
    height = 0.38
    for i, n in enumerate(names):
        y = len(names) - 1 - i
        ax.barh(y + height / 2, ours[n], height=height - 0.04, color=POINTER, label="this model" if i == 0 else None)
        if n in theirs:
            ax.barh(
                y - height / 2,
                theirs[n],
                height=height - 0.04,
                color=BASELINE_COLOR,
                label="typesafe/jev-1.13" if i == 0 else None,
            )
    ax.set_yticks(range(len(names))[::-1], labels, fontsize=8 if row < 0.4 else 9)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.005)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, borderaxespad=0.2)
    ax.set_title(title, loc="left", pad=26)
    fig.savefig(path)
    plt.close(fig)


def accuracy(scores: dict[str, dict]) -> dict[str, float]:
    return {n: s["accuracy"] for n, s in scores.items()}


def per_dataset(scores: dict[str, dict], baseline: dict[str, dict], plots: Path):
    """Test accuracy of every training dataset, grouped by topic."""
    names = sorted(
        (n for n, s in scores.items() if not s["benchmark"]), key=lambda n: (topic(n), -scores[n]["accuracy"])
    )
    labels = [f"{n}  ·  {topic(n)}" for n in names]
    title = "Held-in datasets: test accuracy"
    paired(names, accuracy(scores), accuracy(baseline), labels, title, plots / "test_accuracy.png", 0.32)


def benchmarks(scores: dict[str, dict], baseline: dict[str, dict], plots: Path, trained: bool):
    names = sorted((n for n, s in scores.items() if s["benchmark"]), key=lambda n: -scores[n]["accuracy"])
    title = f"Benchmarks ({'train split trained on' if trained else 'never trained on'}): test accuracy"
    paired(names, accuracy(scores), accuracy(baseline), names, title, plots / "benchmarks.png", 0.42)


def by_topic_vs_baseline(scores: dict[str, dict], baseline: dict[str, dict], benchmark: bool, path: Path):
    """Macro accuracy per topic of both models, over the held-in or the benchmark datasets."""
    part = [n for n, s in scores.items() if s["benchmark"] == benchmark]
    topics = sorted({topic(n) for n in part})
    ours = {t: macro([scores[n] for n in part if topic(n) == t]) for t in topics}
    theirs = {t: macro([baseline[n] for n in part if topic(n) == t and n in baseline]) for t in topics}
    title = f"{'Benchmarks' if benchmark else 'Held-in test'}: macro accuracy by topic"
    paired(topics, ours, theirs, topics, title, path, 0.5, "macro accuracy (mean over datasets)")


def by_topic(scores: dict[str, dict], plots: Path, trained: bool):
    topics = sorted({topic(n) for n in scores})
    held = {t: [s for n, s in scores.items() if topic(n) == t and not s["benchmark"]] for t in topics}
    bench = {t: [s for n, s in scores.items() if topic(n) == t and s["benchmark"]] for t in topics}
    bench_label = "benchmark test (trained on its train split)" if trained else "benchmark"
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(topics) + 1))
    height = 0.38
    for i, t in enumerate(topics):
        y = len(topics) - 1 - i
        if held[t]:
            ax.barh(
                y + height / 2,
                macro(held[t]),
                height=height - 0.04,
                color=POINTER,
                label="held-in test" if not ax.get_legend_handles_labels()[1].count("held-in test") else None,
            )
        if bench[t]:
            ax.barh(
                y - height / 2,
                macro(bench[t]),
                height=height - 0.04,
                color=BENCHMARK,
                label=bench_label if not ax.get_legend_handles_labels()[1].count(bench_label) else None,
            )
    ax.set_yticks(range(len(topics))[::-1], topics)
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro accuracy (mean over datasets)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, borderaxespad=0.2)
    ax.set_title("Accuracy by topic", loc="left", pad=26)
    fig.savefig(plots / "by_topic.png")
    plt.close(fig)


def table(rows: list[tuple[str, dict]], baseline: dict[str, dict] | None = None) -> str:
    head = "| dataset | topic | type | source | n | accuracy | NLL | ECE |"
    sep = "|---|---|---|---|---:|---:|---:|---:|"
    if baseline:
        head += " jev-1.13 acc | jev-1.13 NLL | jev-1.13 ECE |"
        sep += "---:|---:|---:|"
    lines = [head, sep]
    for name, s in rows:
        ds = datasets[name]
        source = f"[{ds.hf_path}](https://huggingface.co/datasets/{ds.hf_path})"
        if ds.hf_name:
            source += f" `{ds.hf_name}`"
        line = (
            f"| {name} | {topic(name)} | {ds.type} | {source} | {s['n']:,} | {s['accuracy']:.3f} | {s['nll']:.3f} "
            f"| {s['ece']:.3f} |"
        )
        if baseline:
            b = baseline.get(name)
            line += f" {b['accuracy']:.3f} | {b['nll']:.3f} | {b['ece']:.3f} |" if b else " - | - | - |"
        lines.append(line)
    return "\n".join(lines)


def main(card: str):
    here = REPO / "cards" / card
    run = REPO / "runs/pointer" / card.removeprefix("pointer-")
    plots = here / "plots"
    plots.mkdir(exist_ok=True)
    # the CLI may have stored the flag as the string "True"
    trained = str(json.loads((run / "config.json").read_text())["train_on_benchmark"]).lower() == "true"
    scores = load_scores(run / "ce-test.json")
    baseline = load_scores(BASELINE)
    training_curve(json.loads((run / "ce/history.json").read_text()), plots)
    per_dataset(scores, baseline, plots)
    benchmarks(scores, baseline, plots, trained)
    by_topic(scores, plots, trained)
    by_topic_vs_baseline(scores, baseline, False, plots / "by_topic_vs_jev.png")

    held = sorted(((n, s) for n, s in scores.items() if not s["benchmark"]), key=lambda x: (topic(x[0]), x[0]))
    bench = sorted(((n, s) for n, s in scores.items() if s["benchmark"]), key=lambda x: (topic(x[0]), x[0]))
    out = ["## Held-in", table(held, baseline), "", "## Benchmarks", table(bench, baseline), ""]
    for label, part in (("held-in", held), ("benchmark", bench)):
        for kind in ("noul", "choice"):
            rows = [s for n, s in part if datasets[n].type == kind]
            if rows:
                out.append(
                    f"- {label} {kind}: {len(rows)} datasets, macro acc {macro(rows):.3f}, "
                    f"NLL {macro(rows, 'nll'):.3f}, ECE {macro(rows, 'ece'):.3f}"
                )
    for label, part in (("held-in", held), ("benchmark", bench)):
        rows = [baseline[n] for n, _ in part if n in baseline]
        out.append(
            f"- jev-1.13 {label}: {len(rows)} datasets, macro acc {macro(rows):.3f}, "
            f"NLL {macro(rows, 'nll'):.3f}, ECE {macro(rows, 'ece'):.3f}"
        )
    (here / "tables.md").write_text("\n".join(out) + "\n")

    # README.md = Hub front matter + the hand-written body.md + the per-dataset tables
    sources = sorted({datasets[n].hf_path for n in scores})
    front = [
        "---",
        "base_model: Qwen/Qwen3.5-2B-Base",
        "library_name: peft",
        "pipeline_tag: text-classification",
        "language:",
        "- en",
        "- ru",
        "tags:",
        "- multiple-choice",
        "- zero-shot-classification",
        "- pointer-network",
        "- lora",
        "- vllm",
        "datasets:",
        *(f"- {source}" for source in sources),
        "---",
        "",
    ]
    tables = [
        "### Held-in (test)",
        "",
        table(held, baseline),
        "",
        "### Benchmarks (test)",
        "",
        table(bench, baseline),
        "",
    ]
    body = (here / "body.md").read_text()
    (here / "README.md").write_text("\n".join(front) + body + "\n".join(tables))


if __name__ == "__main__":
    main(sys.argv[1])
