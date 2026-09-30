
"""Two runs side by side, per dataset. Works on the {dataset: metrics} dicts that `evaluate`
returns, so any two registered models can be compared."""

import json
from pathlib import Path

# (key, label, whether a higher value is better)
METRICS = (
    ("accuracy", "accuracy", True),
    ("f1", "macro F1", True),
    ("precision", "macro precision", True),
    ("recall", "macro recall", True),
    ("nll", "NLL", False),
    ("ece", "ECE", False),
)
HIGHER_IS_BETTER = {key: higher for key, _, higher in METRICS}
MACRO = "macro"
TEMPLATE = Path(__file__).parent / "compare.html"

Metrics = dict[str, dict[str, float]]


def shared(a: Metrics, b: Metrics) -> tuple[Metrics, Metrics, list[str]]:
    """Both runs cut to the datasets they have in common, with `macro` taken again over those
    alone: each run's own macro covers its own datasets, so the two would not be comparable.
    Also returns the datasets only one run has."""
    names = [name for name in a if name in b and name != MACRO]
    if not names:
        raise ValueError("the runs have no dataset in common")
    left_out = sorted((set(a) ^ set(b)) - {MACRO})

    def cut(metrics: Metrics) -> Metrics:
        rows = {name: metrics[name] for name in names}
        macro = {key: sum(row[key] for row in rows.values()) / len(rows) for key in HIGHER_IS_BETTER}
        return rows | {MACRO: macro | {"n": sum(row["n"] for row in rows.values())}}

    return cut(a), cut(b), left_out


def deltas(a: Metrics, b: Metrics, metric: str) -> list[tuple[str, float, float, float]]:
    """(dataset, a, b, b - a) on `metric`, the largest gain of `b` first and `macro` last.
    `a` and `b` come from `shared`."""
    rows = [(name, a[name][metric], b[name][metric], b[name][metric] - a[name][metric]) for name in a if name != MACRO]
    rows.sort(key=lambda row: row[3], reverse=HIGHER_IS_BETTER[metric])
    return rows + [(MACRO, a[MACRO][metric], b[MACRO][metric], b[MACRO][metric] - a[MACRO][metric])]


def table(a: Metrics, b: Metrics, labels: tuple[str, str], metric: str) -> str:
    """One line per dataset: both values, the difference, and which run is better."""
    rows = deltas(a, b, metric)
    higher = HIGHER_IS_BETTER[metric]
    lines = [f"{metric + (' (higher is better)' if higher else ' (lower is better)'):32} {'n':>6}  {'A':>7}  {'B':>7}  {'B - A':>7}"]
    for name, left, right, delta in rows:
        better = "" if delta == 0 else "B" if (delta > 0) == higher else "A"
        lines.append(f"{name:32} {a[name]['n']:>6}  {left:>7.3f}  {right:>7.3f}  {delta:>+7.3f}  {better}")
    wins = sum((delta > 0) == higher for _, _, _, delta in rows[:-1] if delta != 0)
    ties = sum(delta == 0 for _, _, _, delta in rows[:-1])
    lines.append(f"A = {labels[0]}\nB = {labels[1]}\nB is better on {wins} of {len(rows) - 1} datasets, A on "
                 f"{len(rows) - 1 - wins - ties}")
    return "\n".join(lines)


def write_html(a: Metrics, b: Metrics, labels: tuple[str, str], partition: str, metric: str, path: Path):
    """A standalone page: a dumbbell chart of both runs per dataset, a metric picker and the table."""
    data = {
        "labels": labels,
        "partition": partition,
        "metric": metric,
        "metrics": [{"key": key, "label": label, "higher": higher} for key, label, higher in METRICS],
        "datasets": [name for name in a if name != MACRO],
        "a": a,
        "b": b,
    }
    # `</` would end the script block the data sits in
    payload = json.dumps(data).replace("</", "<\\/")
    path.write_text(TEMPLATE.read_text().replace("__DATA__", payload))
