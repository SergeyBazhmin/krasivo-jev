from pathlib import Path
from typing import Annotated

import typer

from jev_datasets import datasets
from jev_datasets.constants import MAX_SAMPLES, ROOT_DIR, RU_DIR

app = typer.Typer(help="Build the jev datasets from the Hub.", no_args_is_help=True)


@app.command("list")
def list_datasets():
    """Print the available dataset names."""
    typer.echo("\n".join(sorted(datasets)))


@app.command()
def prepare(
    names: Annotated[list[str], typer.Argument(help="datasets to build; `all` for every one")],
    max_samples: Annotated[int, typer.Option(help="per-split cap")] = MAX_SAMPLES,
    output_dir: Annotated[Path, typer.Option(help="saved to OUTPUT_DIR/NAME")] = ROOT_DIR,
):
    """Load datasets, prepare them and save them to disk."""
    names = sorted(datasets) if names == ["all"] else names
    if unknown := [name for name in names if name not in datasets]:
        raise typer.BadParameter(f"unknown datasets: {', '.join(unknown)} (see `list`)", param_hint="NAMES")

    for name in names:
        dataset = datasets[name]
        typer.echo(f"[{name}] loading {dataset.hf_path}" + (f" ({dataset.hf_name})" if dataset.hf_name else ""))
        dataset.prepare()
        dataset.add_type()
        dataset.limit(max_samples)
        dataset.data.save_to_disk(output_dir / name)
        typer.echo(f"[{name}] saved {dict(dataset.data.num_rows)} to {output_dir / name}")


@app.command()
def translate(
    names: Annotated[list[str], typer.Argument(help="saved datasets to translate; `all` for every one in INPUT_DIR")],
    input_dir: Annotated[Path, typer.Option(help="where `prepare` saved them")] = ROOT_DIR,
    output_dir: Annotated[Path, typer.Option(help="saved to OUTPUT_DIR/NAME")] = RU_DIR,
    url: Annotated[str, typer.Option(help="OpenAI-compatible API base")] = "http://localhost:8000/v1",
    model: Annotated[str | None, typer.Option(help="served model; the first one the server lists by default")] = None,
    concurrency: Annotated[int, typer.Option(help="requests in flight")] = 64,
):
    """Translate saved datasets to Russian with a local vLLM (or any OpenAI-compatible) server."""
    from jev_datasets.translate import Translator, translate_saved

    built = sorted(p.parent.name for p in input_dir.glob("*/dataset_dict.json"))
    names = built if names == ["all"] else names
    if unknown := [name for name in names if name not in built]:
        raise typer.BadParameter(f"not saved under {input_dir}: {', '.join(unknown)}", param_hint="NAMES")

    translator = Translator(url, model, output_dir / "translations.jsonl", concurrency)
    for name in names:
        translate_saved(name, input_dir, output_dir, translator)


if __name__ == "__main__":
    app()
