"""Any number of runs in one report: a leaderboard, per-dataset tables, wins between runs, scores by topic
and dataset type, and the datasets that split the runs or that none solves. Built on the {dataset: metrics}
dicts that `evaluate` returns, cut to common datasets by `compare.shared_all`."""

import json
import math
import statistics
from pathlib import Path

from jev_model.compare import HIGHER_IS_BETTER, MACRO, METRICS, Metrics, shared_all

TEMPLATE = Path(__file__).parent / "report.html"
LABELS = {key: label for key, label, _ in METRICS}
# datasets in the spread and hardest tables
TOP = 10


def value(metrics: Metrics, name: str, key: str) -> float | None:
    """None when the run lacks the dataset or the metric is NaN (generation gives no probabilities)."""
    v = metrics.get(name, {}).get(key)
    return None if v is None or math.isnan(v) else v


def dataset_info() -> dict[str, tuple[str, str]]:
    """{name: (topic, type)} from the dataset registry; the topic is the subpackage of the converter."""
    try:
        from jev_datasets import datasets
    except ImportError:
        return {}
    return {name: (type(ds).__module__.split(".")[1], ds.type.value) for name, ds in datasets.items()}


def better(a: float, b: float, key: str) -> bool:
    return a > b if HIGHER_IS_BETTER[key] else a < b


def ranks(values: list[float | None], key: str) -> list[float | None]:
    """Rank 1 for the best value, ties sharing their mean rank; None stays unranked."""
    present = [v for v in values if v is not None]
    result = []
    for v in values:
        if v is None:
            result.append(None)
            continue
        ahead = sum(better(o, v, key) for o in present)
        tied = sum(o == v for o in present)
        result.append(ahead + (tied + 1) / 2)
    return result


def analyse(runs: list[Metrics], labels: list[str], common: list[str], info: dict, key: str) -> dict:
    """Everything the page shows for one metric, over the datasets every run has."""
    higher = HIGHER_IS_BETTER[key]
    k = len(runs)
    table = {name: [value(run, name, key) for run in runs] for name in common}
    if all(v is None for row in table.values() for v in row):
        return {"available": False}

    rank_rows = {name: ranks(table[name], key) for name in common}

    def best_of(name: str) -> tuple[float, int] | None:
        present = [(v, i) for i, v in enumerate(table[name]) if v is not None]
        return (max(present) if higher else min(present)) if present else None

    tops = {name: best_of(name) for name in common}
    leaderboard = []
    for i in range(k):
        mine = [table[n][i] for n in common if table[n][i] is not None]
        mine_ranks = [rank_rows[n][i] for n in common if rank_rows[n][i] is not None]
        leaderboard.append(
            {
                "run": i,
                "macro": value(runs[i], MACRO, key),
                "median": statistics.median(mine) if mine else None,
                "mean_rank": sum(mine_ranks) / len(mine_ranks) if mine_ranks else None,
                "best": sum(tops[n] is not None and table[n][i] == tops[n][0] for n in common),
            }
        )
    missing = math.inf if higher else -math.inf
    leaderboard.sort(key=lambda r: missing if r["macro"] is None else r["macro"], reverse=higher)

    wins = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            pairs = [(table[n][i], table[n][j]) for n in common if None not in (table[n][i], table[n][j])]
            if i == j or not pairs:
                continue
            wins[i][j] = {
                "wins": sum(better(a, b, key) for a, b in pairs),
                "losses": sum(better(b, a, key) for a, b in pairs),
                "total": len(pairs),
                "delta": sum(a - b for a, b in pairs) / len(pairs),
            }

    def groups(index: int) -> dict[str, list[float | None]]:
        members: dict[str, list[str]] = {}
        for name in common:
            members.setdefault(info.get(name, ("other", "other"))[index], []).append(name)
        result = {}
        for group, names in sorted(members.items()):
            result[group] = {"n": len(names), "values": []}
            for i in range(k):
                vs = [table[n][i] for n in names if table[n][i] is not None]
                result[group]["values"].append(sum(vs) / len(vs) if vs else None)
        return result

    spread = []
    for name in (n for n in common if tops[n] is not None):
        present = [v for v in table[name] if v is not None]
        top, i = tops[name]
        spread.append({"dataset": name, "spread": max(present) - min(present), "best": top, "best_run": i})
    spread.sort(key=lambda r: r["spread"], reverse=True)
    hardest = sorted(spread, key=lambda r: r["best"], reverse=not higher)

    result = {
        "available": True,
        "leaderboard": leaderboard,
        "wins": wins,
        "topics": groups(0),
        "types": groups(1),
        "spread": spread[:TOP],
        "hardest": hardest[:TOP],
    }
    result["findings"] = findings(result, labels, common, key)
    return result


def fmt(v: float) -> str:
    return f"{v:.3f}"


def findings(analysis: dict, labels: list[str], common: list[str], key: str) -> list[str]:
    """Plain sentences from the numbers, the most telling first."""
    label = LABELS[key]
    board = [r for r in analysis["leaderboard"] if r["macro"] is not None]
    out = []
    if board:
        lead = board[0]
        line = f"{labels[lead['run']]} leads on macro {label} ({fmt(lead['macro'])})"
        if len(board) > 1:
            gap = abs(lead["macro"] - board[1]["macro"])
            line += f", {fmt(gap)} ahead of {labels[board[1]['run']]}"
        out.append(line + ".")
    by_best = max(analysis["leaderboard"], key=lambda r: r["best"])
    out.append(
        f"{labels[by_best['run']]} is best on {by_best['best']} of {len(common)} datasets (ties count for each)."
    )
    ranked = [r for r in analysis["leaderboard"] if r["mean_rank"] is not None]
    if ranked:
        top = min(ranked, key=lambda r: r["mean_rank"])
        if top["run"] != (board[0]["run"] if board else None):
            out.append(f"{labels[top['run']]} has the best mean rank ({top['mean_rank']:.2f}) despite a lower macro.")

    leaders: dict[int, list[str]] = {}
    for topic, group in analysis["topics"].items():
        present = [(v, i) for i, v in enumerate(group["values"]) if v is not None]
        if present and len(labels) > 1:
            _, i = max(present) if HIGHER_IS_BETTER[key] else min(present)
            leaders.setdefault(i, []).append(topic)
    if len(leaders) > 1:
        parts = [f"{labels[i]} on {', '.join(topics)}" for i, topics in leaders.items()]
        out.append("By topic, the lead is split: " + "; ".join(parts) + ".")
    for r in analysis["leaderboard"]:
        topics = [
            (g["values"][r["run"]], t) for t, g in analysis["topics"].items() if g["values"][r["run"]] is not None
        ]
        if len(topics) > 1:
            worst = min(topics) if HIGHER_IS_BETTER[key] else max(topics)
            out.append(f"Weakest topic of {labels[r['run']]}: {worst[1]} ({fmt(worst[0])}).")

    if len(labels) > 1 and analysis["spread"]:
        s = analysis["spread"][0]
        out.append(
            f"The runs differ most on {s['dataset']} (spread {fmt(s['spread'])}, best: {labels[s['best_run']]})."
        )
    if analysis["hardest"]:
        h = analysis["hardest"][0]
        out.append(f"Hardest dataset: {h['dataset']}, where the best run reaches only {fmt(h['best'])}.")
    return out


def clean(obj):
    """NaN as null, so the payload is valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    return obj


def build(runs: list[Metrics], labels: list[str], partition: str, metric: str, union: bool) -> tuple[dict, list[str]]:
    """The page payload, and the datasets some run lacks."""
    runs, left_out = shared_all(runs, union)
    common = [name for name in runs[0] if name != MACRO and name not in left_out]
    names = common + (left_out if union else [])
    info = dataset_info()
    payload = {
        "labels": labels,
        "partition": partition,
        "metric": metric,
        "metrics": [{"key": key, "label": label, "higher": higher} for key, label, higher in METRICS],
        "datasets": [
            {
                "name": name,
                "topic": info.get(name, ("other", "other"))[0],
                "type": info.get(name, ("other", "other"))[1],
                "n": max(run[name]["n"] for run in runs if name in run),
                "common": name in common,
            }
            for name in names
        ],
        "values": {name: [run.get(name) for run in runs] for name in names + [MACRO]},
        "analysis": {key: analyse(runs, labels, common, info, key) for key in HIGHER_IS_BETTER},
    }
    return clean(payload), left_out


def leaderboard_table(payload: dict) -> str:
    """The leaderboard of the chosen metric as printed by `report`."""
    key = payload["metric"]
    analysis = payload["analysis"][key]
    if not analysis["available"]:
        return f"no run has {key}"
    width = max(map(len, payload["labels"])) + 2
    lines = [f"{'run':{width}} {'macro ' + key:>16}  {'mean rank':>9}  {'best on':>7}  {'median':>7}"]
    for r in analysis["leaderboard"]:
        cells = [
            "n/a" if r["macro"] is None else fmt(r["macro"]),
            "n/a" if r["mean_rank"] is None else f"{r['mean_rank']:.2f}",
            str(r["best"]),
            "n/a" if r["median"] is None else fmt(r["median"]),
        ]
        lines.append(
            f"{payload['labels'][r['run']]:{width}} {cells[0]:>16}  {cells[1]:>9}  {cells[2]:>7}  {cells[3]:>7}"
        )
    return "\n".join(lines + [""] + [f"- {line}" for line in analysis["findings"]])


def write_html(payload: dict, path: Path):
    # `</` would end the script block the data sits in
    data = json.dumps(payload).replace("</", "<\\/")
    path.write_text(TEMPLATE.read_text().replace("__DATA__", data))
