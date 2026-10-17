import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import datasets as hf_datasets
import typer

from jev_datasets import datasets
from jev_datasets.constants import MAX_SAMPLES, ROOT_DIR

app = typer.Typer(help="Build the jev datasets from the Hub.", no_args_is_help=True)


@app.command("list")
def list_datasets():
    """Print the available dataset names."""
    typer.echo("\n".join(sorted(datasets)))


def build(name: str, max_samples: int, output_dir: Path, quiet: bool = False):
    """Loads one dataset, prepares it and saves it to OUTPUT_DIR/NAME."""
    if quiet:
        # bars from several workers would interleave into noise
        hf_datasets.disable_progress_bars()
    dataset = datasets[name]
    typer.echo(f"[{name}] loading {dataset.hf_path}" + (f" ({dataset.hf_name})" if dataset.hf_name else ""))
    dataset.prepare()
    dataset.regroup()
    dataset.add_type()
    dataset.limit(max_samples)
    dataset.save_to(output_dir / name)
    typer.echo(f"[{name}] saved {dict(dataset.data.num_rows)} to {output_dir / name}")


@app.command()
def prepare(
    names: Annotated[list[str], typer.Argument(help="datasets to build; `all` for every one")],
    max_samples: Annotated[int, typer.Option(help="per-split cap")] = MAX_SAMPLES,
    output_dir: Annotated[Path, typer.Option(help="saved to OUTPUT_DIR/NAME")] = ROOT_DIR,
    jobs: Annotated[int, typer.Option("--jobs", "-j", min=1, help="datasets built at once, one process each")] = 1,
):
    """Load datasets, prepare them and save them to disk."""
    names = sorted(datasets) if names == ["all"] else names
    if unknown := [name for name in names if name not in datasets]:
        raise typer.BadParameter(f"unknown datasets: {', '.join(unknown)} (see `list`)", param_hint="NAMES")

    if jobs == 1 or len(names) == 1:
        for name in names:
            build(name, max_samples, output_dir)
        return

    # One fresh process per dataset, so a finished build's memory goes back to the system; one failure doesn't
    # stop the others, and the failed names are listed at the end.
    failed = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(min(jobs, len(names)), mp_context=context, max_tasks_per_child=1) as pool:
        futures = {pool.submit(build, name, max_samples, output_dir, True): name for name in names}
        for future in as_completed(futures):
            if error := future.exception():
                failed.append(futures[future])
                typer.echo(f"[{futures[future]}] failed: {error!r}", err=True)
    if failed:
        typer.echo(f"failed: {', '.join(sorted(failed))}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
