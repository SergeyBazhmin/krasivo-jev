import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tqdm import tqdm

from jev_model import models
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
        raise typer.BadParameter(str(error), param_hint="--config/--set")


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
def show_config(model_name: Annotated[str, typer.Argument(metavar="MODEL")], config: ConfigFile = None,
                overrides: Overrides = None):
    """Print a model's resolved config as JSON; save it to start a config file."""
    model = get_model(model_name)
    typer.echo(dump(resolve_config(model, path=config, overrides=overrides or [])))


@app.command()
def train(
    model_name: Annotated[str, typer.Argument(metavar="MODEL")],
    config: ConfigFile = None,
    overrides: Overrides = None,
    stages: Annotated[list[str] | None, typer.Option("--stage", help="stages to run; the model's default ones if unset")] = None,
    run_dir: Annotated[Path | None, typer.Option(help="RUNS_DIR/MODEL/<timestamp> by default. An existing run "
                                                "keeps its config, so later stages can be added to it")] = None,
    device: Device = "",
):
    """Run a model's training pipeline, or some of its stages."""
    model = get_model(model_name)
    if unknown := [s for s in stages or [] if s not in model.stages]:
        raise typer.BadParameter(f"{model.name} has no stages {unknown}; it has {list(model.stages)}", param_hint="--stage")
    run_dir = run_dir or RUNS_DIR / model.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    previous = read_run(run_dir)
    if previous and previous["model"] != model.name:
        raise typer.BadParameter(f"{run_dir} belongs to {previous['model']}", param_hint="--run-dir")
    base = read_file(run_dir / CONFIG_FILE) if previous else None
    resolved = resolve_config(model, base=base, path=config, overrides=overrides or [])
    model.train(resolved, run_dir, default_device(device), stages)
    typer.echo(f"run: {run_dir}")


@app.command("eval")
def eval_command(
    run_dir: Annotated[Path, typer.Argument(help="directory written by `train`")],
    partition: str = "test",
    stage: Stage = None,
    device: Device = "",
):
    """Per-dataset accuracy, NLL and ECE; saved to RUN_DIR/<stage>-<partition>.json."""
    model, config = open_run(run_dir)
    metrics = model.evaluate(config, run_dir, partition, default_device(device), stage)
    for name, m in metrics.items():
        typer.echo(f"{name:32} n={m['n']:>6}  acc={m['accuracy']:.3f}  nll={m['nll']:.3f}  ece={m['ece']:.3f}")
    (run_dir / f"{stage or 'last'}-{partition}.json").write_text(json.dumps(metrics, indent=2))


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
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)]))


if __name__ == "__main__":
    app()
