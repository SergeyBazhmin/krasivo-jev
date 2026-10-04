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

    typer.echo(table(run_metrics(run_dir, partition, stage, device, fresh=True)))


def run_metrics(
    run_dir: Path, partition: str, stage: str | None, device: str, fresh: bool
) -> dict[str, dict[str, float]]:
    """A run's {dataset: metrics}, read from RUN_DIR/<stage>-<partition>.json when `eval` has
    already written it, unless `fresh`."""
    path = run_dir / f"{stage or 'last'}-{partition}.json"
    if path.exists() and not fresh:
        return json.loads(path.read_text())
    model, config = open_run(run_dir)
    metrics = model.evaluate(config, run_dir, partition, default_device(device), stage)
    path.write_text(json.dumps(metrics, indent=2))
    return metrics


@app.command()
def compare(
    run_a: Annotated[Path, typer.Argument(help="run A, the baseline")],
    run_b: Annotated[Path, typer.Argument(help="run B")],
    partition: str = "test",
    metric: Annotated[str, typer.Option(help="accuracy, f1, precision, recall, nll or ece")] = "accuracy",
    stage_a: Annotated[str | None, typer.Option(help="stage of run A; the last trained by default")] = None,
    stage_b: Annotated[str | None, typer.Option(help="stage of run B; the last trained by default")] = None,
    output: Annotated[
        Path | None, typer.Option(help="HTML page with the chart; RUN_B/compare-<run A>-<partition>.html by default")
    ] = None,
    fresh: Annotated[bool, typer.Option(help="evaluate again instead of reading the scores `eval` saved")] = False,
    device: Device = "",
):
    """Compare two runs per dataset: a table here and a chart in an HTML page. The runs may be of
    different models; only the datasets both were trained on are compared."""
    from jev_model import compare as comparison

    if metric not in comparison.HIGHER_IS_BETTER:
        raise typer.BadParameter(
            f"unknown metric {metric!r}; one of {list(comparison.HIGHER_IS_BETTER)}", param_hint="--metric"
        )
    labels = tuple(
        f"{read_run(run).get('model', '?')} {run.resolve().name}" + (f" ({stage})" if stage else "")
        for run, stage in ((run_a, stage_a), (run_b, stage_b))
    )
    try:
        a, b, left_out = comparison.shared(
            run_metrics(run_a, partition, stage_a, device, fresh), run_metrics(run_b, partition, stage_b, device, fresh)
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="RUN_B") from error
    typer.echo(comparison.table(a, b, labels, metric))
    if left_out:
        typer.echo(f"in one run only, not compared: {', '.join(left_out)}")
    output = output or run_b / f"compare-{run_a.resolve().name}-{partition}.html"
    comparison.write_html(a, b, labels, partition, metric, output)
    typer.echo(f"chart: {output}")


def stage_metrics(run_dir: Path, partition: str, stage: str, last: bool, device: str, fresh: bool) -> dict:
    """`run_metrics` of one stage. The last stage falls back to the scores `eval` saved without --stage
    (RUN/last-<partition>.json), which are the same ones, before evaluating again."""
    fallback = run_dir / f"last-{partition}.json"
    if last and not fresh and not (run_dir / f"{stage}-{partition}.json").exists() and fallback.exists():
        return json.loads(fallback.read_text())
    return run_metrics(run_dir, partition, stage, device, fresh)


@app.command()
def report(
    runs: Annotated[list[Path], typer.Argument(help="run directories written by `train`")],
    partition: str = "test",
    metric: Annotated[str, typer.Option(help="metric the page opens on and the terminal shows")] = "accuracy",
    stages: Annotated[
        list[str] | None, typer.Option("--stage", help="only these stages; every finished stage by default")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="HTML page; ./report-<partition>.html by default")] = None,
    all_datasets: Annotated[
        bool, typer.Option(help="also list datasets some run lacks; macro still covers the common ones only")
    ] = False,
    fresh: Annotated[bool, typer.Option(help="evaluate again instead of reading the scores `eval` saved")] = False,
    device: Device = "",
):
    """A report over every finished stage of the given runs: leaderboard, per-dataset tables, wins between them,
    scores by topic and dataset type, and the datasets that split them. Each stage is scored from
    RUN/<stage>-<partition>.json, evaluated first when missing. The leaderboard prints here, the rest is an HTML page."""
    from jev_model import report as reporting
    from jev_model.compare import HIGHER_IS_BETTER

    if metric not in HIGHER_IS_BETTER:
        raise typer.BadParameter(f"unknown metric {metric!r}; one of {list(HIGHER_IS_BETTER)}", param_hint="--metric")
    labels, metrics = [], []
    for run in runs:
        info = read_run(run)
        if "model" not in info:
            raise typer.BadParameter(f"{run} is not a run directory", param_hint="RUNS")
        model = get_model(info["model"])
        scored = [stage for stage in info.get("stages", []) if stage in (model.eval_stages or model.stages)]
        done = [stage for stage in scored if not stages or stage in stages]
        if not done:
            typer.echo(f"{run}: no finished stage{' among ' + ', '.join(stages) if stages else ''}, skipped")
            continue
        for stage in done:
            labels.append(f"{info['model']} {run.resolve().name} / {stage}")
            metrics.append(stage_metrics(run, partition, stage, stage == scored[-1], device, fresh))
    if not metrics:
        raise typer.BadParameter("no finished stage to report", param_hint="RUNS")
    try:
        payload, left_out = reporting.build(metrics, labels, partition, metric, all_datasets)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="RUNS") from error
    typer.echo(reporting.leaderboard_table(payload))
    if left_out:
        typer.echo(f"\nnot in every run{', listed apart' if all_datasets else ', left out'}: {', '.join(left_out)}")
    output = output or Path(f"report-{partition}.html")
    reporting.write_html(payload, output)
    typer.echo(f"report: {output}")


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
def ui(port: int = 8501):
    """Open a local page to try trained runs by hand (needs the `ui` extra)."""
    import subprocess
    import sys

    script = Path(__file__).parent / "ui" / "app.py"
    raise typer.Exit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)])
    )


if __name__ == "__main__":
    app()
