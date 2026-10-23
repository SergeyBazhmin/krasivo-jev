import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from jev_model import distributed, models
from jev_model.base import CONFIG_FILE, JevModel, read_run
from jev_model.config import dump, load_config, read_file
from jev_model.constants import RUNS_DIR

# # log through tqdm so messages print above an active progress bar instead of breaking it
# logger.remove()
# logger.add(lambda message: tqdm.write(message, end=""), colorize=True)

app = typer.Typer(help="Train and evaluate models on the jev datasets.", no_args_is_help=True)

ConfigFile = Annotated[Path | None, typer.Option("--config", "-c", help="JSON or TOML config; see `config MODEL`")]
Overrides = Annotated[list[str] | None, typer.Option("--set", "-s", help="override a config key: `ce.lr=3e-4`")]
Device = Annotated[str, typer.Option(help="torch device; cuda when available")]
Stage = Annotated[str | None, typer.Option(help="stage whose weights to use; the last trained by default")]


def get_model(name: str) -> JevModel:
    if name not in models:
        raise typer.BadParameter(f"unknown model {name!r} (see `list`)", param_hint="MODEL")
    return models[name]


def default_device(device: str) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_config(model: JevModel, **kwargs):
    try:
        return load_config(model.config_class, **kwargs)
    except (KeyError, ValueError, TypeError) as error:
        raise typer.BadParameter(str(error), param_hint="--config/--set") from error


def open_run(run_dir: Path):
    """The model and config a run was trained with."""
    info = read_run(run_dir)
    if "model" not in info:
        raise typer.BadParameter(f"{run_dir} is not a run directory", param_hint="RUN_DIR")
    model = get_model(info["model"])
    return model, resolve_config(model, base=read_file(run_dir / CONFIG_FILE))


@app.command("list")
def list_models():
    """Print the registered models and their pipeline stages."""
    for name, model in sorted(models.items()):
        stages = " -> ".join(f"{s}*" if s in model.default_stages else s for s in model.stages)
        typer.echo(f"{name:24} {stages:24} {model.description}")
    typer.echo("\n* runs by default")


@app.command("config")
def show_config(
    model_name: Annotated[str, typer.Argument(metavar="MODEL")], config: ConfigFile = None, overrides: Overrides = None
):
    """Print a model's resolved config as JSON; save it to start a config file."""
    model = get_model(model_name)
    typer.echo(dump(resolve_config(model, path=config, overrides=overrides or [])))


@app.command()
def train(
    model_name: Annotated[str, typer.Argument(metavar="MODEL")],
    config: ConfigFile = None,
    overrides: Overrides = None,
    stages: Annotated[
        list[str] | None, typer.Option("--stage", help="stages to run; the model's default ones if unset")
    ] = None,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            help="RUNS_DIR/MODEL/<timestamp> by default. An existing run "
            "keeps its config, so later stages can be added to it"
        ),
    ] = None,
    device: Device = "",
):
    """Run a model's training pipeline, or some of its stages. Under `accelerate launch -m jev_model.cli train ...`
    a model that supports it trains on every GPU with DDP."""
    model = get_model(model_name)
    if unknown := [s for s in stages or [] if s not in model.stages]:
        raise typer.BadParameter(
            f"{model.name} has no stages {unknown}; it has {list(model.stages)}", param_hint="--stage"
        )
    device = default_device(device)
    run_dir = distributed.setup(
        run_dir or RUNS_DIR / model.name / datetime.now().strftime("%Y%m%d-%H%M%S"), cpu=device == "cpu"
    )
    previous = read_run(run_dir)
    if previous and previous["model"] != model.name:
        raise typer.BadParameter(f"{run_dir} belongs to {previous['model']}", param_hint="--run-dir")
    base = read_file(run_dir / CONFIG_FILE) if previous else None
    resolved = resolve_config(model, base=base, path=config, overrides=overrides or [])
    model.train(resolved, run_dir, device, stages)
    if distributed.is_main():
        typer.echo(f"run: {run_dir}")


@app.command("eval")
def eval_command(
    run_dir: Annotated[Path, typer.Argument(help="directory written by `train`")],
    partition: str = "test",
    stage: Stage = None,
    device: Device = "",
):
    """Per-dataset accuracy, NLL and ECE; saved to RUN_DIR/<stage>-<partition>.json."""
    from jev_model.metrics import table

    model, config = open_run(run_dir)
    metrics = model.evaluate(config, run_dir, partition, default_device(device), stage)
    (run_dir / f"{stage or 'last'}-{partition}.json").write_text(json.dumps(metrics, indent=2))
    typer.echo(table(metrics))


@app.command()
def thresholds(
    run_dir: Annotated[Path, typer.Argument(help="directory written by `train`, evaluated with `eval`")],
    partition: str = "test",
    stage: Stage = None,
    datasets: Annotated[
        list[str] | None, typer.Option("--dataset", help="noul datasets to score; every saved one by default")
    ] = None,
    cutoffs: Annotated[str, typer.Option("--thresholds", help="comma-separated probability cut-offs")] = ",".join(
        f"{t / 20:g}" for t in range(1, 20)
    ),
    positive: Annotated[str, typer.Option(help="the class to detect: `yes` or `no`")] = "yes",
):
    """Precision, recall and F1 of a noul dataset's `yes` (or `no`) answer at each probability cut-off, from the
    predictions `eval` saved; nothing is run. Saved to RUN_DIR/<stage>-<partition>-thresholds.json."""
    import torch

    from jev_model.data import YES, kind
    from jev_model.metrics import average_precision, threshold_scores

    if positive not in ("yes", "no"):
        raise typer.BadParameter("must be `yes` or `no`", param_hint="--positive")
    info = read_run(run_dir)
    if "model" not in info:
        raise typer.BadParameter(f"{run_dir} is not a run directory", param_hint="RUN_DIR")
    stage = stage or (info.get("stages") or [None])[-1]
    files = {
        p.name.removeprefix(f"{partition}-").removesuffix(".jsonl"): p
        for p in (run_dir / str(stage)).glob(f"{partition}-*.jsonl")
    }
    if not files:
        raise typer.BadParameter(f"no {partition} predictions in {run_dir / str(stage)}; run `jev-model eval` first")
    if missing := [name for name in datasets or [] if name not in files]:
        raise typer.BadParameter(f"no saved {partition} predictions for {', '.join(missing)}", param_hint="--dataset")
    cuts = sorted(float(t) for t in cutoffs.split(","))

    results = {}
    for name in datasets or sorted(files):
        rows = [json.loads(line) for line in files[name].read_text().splitlines()]
        if not rows or kind(rows[0]["options"]) != "noul":
            if datasets:
                raise typer.BadParameter(f"{name} is not a noul dataset", param_hint="--dataset")
            continue
        # rows the API rejected carry no probabilities
        answered = [r for r in rows if r["probs"] is not None]
        index = [
            next(i for i, text in enumerate(r["options"]) if (text.strip().lower() in YES) == (positive == "yes"))
            for r in answered
        ]
        p = torch.tensor([r["probs"][i] for r, i in zip(answered, index)])
        # a soft label counts as positive when it puts at least half its mass there, as the argmax does
        gold = torch.tensor([r["label"][i] >= 0.5 for r, i in zip(answered, index)])
        sweep = threshold_scores(p, gold, cuts)
        best = max((s for s in sweep if s["f1"] is not None), key=lambda s: s["f1"], default=None)
        results[name] = {
            "n": len(answered),
            "skipped": len(rows) - len(answered),
            "positive": positive,
            "base_rate": gold.float().mean().item(),
            "average_precision": average_precision(p, gold),
            "best_f1_threshold": best and best["threshold"],
            "thresholds": sweep,
        }

    if not results:
        raise typer.BadParameter(f"no noul datasets among the saved {partition} predictions", param_hint="RUN_DIR")
    (run_dir / f"{stage}-{partition}-thresholds.json").write_text(json.dumps(results, indent=2))

    def fmt(x: float | None) -> str:
        return "    -" if x is None else f"{x:.3f}"

    for name, r in results.items():
        typer.echo(
            f"\n{name}  n={r['n']}  {positive} rate={r['base_rate']:.3f}  AP={fmt(r['average_precision'])}"
            f"  best F1 at {r['best_f1_threshold']}"
            + (f"  ({r['skipped']} unanswered skipped)" if r["skipped"] else "")
        )
        typer.echo("  cut-off  precision  recall     f1  predicted+")
        for s in r["thresholds"]:
            typer.echo(
                f"  {s['threshold']:>7g}  {fmt(s['precision']):>9}  {fmt(s['recall']):>6}  {fmt(s['f1']):>5}"
                f"  {s['positive_rate']:>10.3f}"
            )


@app.command()
def predict(
    run_dir: Annotated[Path, typer.Argument(help="directory written by `train`")],
    question: Annotated[str, typer.Option()],
    options: Annotated[list[str], typer.Option("--option", help="repeat for each option")],
    state: str = "",
    stage: Stage = None,
    device: Device = "",
):
    """Probabilities over the options of one question."""
    model, config = open_run(run_dir)
    (probs,) = model.predictor(config, run_dir, default_device(device), stage).predict([(state, question, options)])
    for option, p in zip(options, probs):
        typer.echo(f"{p:.3f}  {option}")


@app.command()
def export(
    run_dir: Annotated[Path, typer.Argument(help="directory written by `train`")],
    out: Annotated[Path, typer.Option(help="folder to write the merged model to")],
    stage: Stage = None,
    device: Device = "",
):
    """Merge a run into one folder that vLLM serves (models that support it; see `jev_vllm`)."""
    model, _ = open_run(run_dir)
    if not callable(getattr(model, "export", None)):
        raise typer.BadParameter(f"{model.name} has no export", param_hint="RUN_DIR")
    model.export(run_dir, out, default_device(device), stage)
    typer.echo(f"exported: {out}")


@app.command()
def push(
    export_dir: Annotated[Path, typer.Argument(help="folder written by `export`", file_okay=False)],
    repo: Annotated[str, typer.Option(help="Hugging Face repo id, `user/name`")],
    private: bool = False,
):
    """Upload an export to the Hub as it is: vLLM serves the repo, and `JevHolder.from_pretrained` loads it."""
    from huggingface_hub import HfApi

    from jev_model.pointer.model import HEAD_FILE

    missing = [name for name in ("config.json", HEAD_FILE) if not (export_dir / name).is_file()]
    if missing:
        raise typer.BadParameter(
            f"no {', '.join(missing)} in {export_dir}; run `export` first", param_hint="EXPORT_DIR"
        )
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=export_dir)
    typer.echo(f"pushed: https://huggingface.co/{repo}")


if __name__ == "__main__":
    app()
