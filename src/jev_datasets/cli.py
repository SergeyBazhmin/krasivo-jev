from pathlib import Path
from typing import Annotated

import typer

from jev_datasets import datasets
from jev_datasets.constants import MAX_SAMPLES, ROOT_DIR

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
        dataset.limit(max_samples)
        dataset.data.save_to_disk(output_dir / name)
        typer.echo(f"[{name}] saved {dict(dataset.data.num_rows)} to {output_dir / name}")


if __name__ == "__main__":
    app()
