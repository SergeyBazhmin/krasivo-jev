"""`uv run serve MODEL`: build the vLLM image with this package and start it on a `jev-model export` folder or a Hub
repo pushed from one (`jev-model push`). Only needs Docker on the host; vLLM itself runs in the container."""

import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer

IMAGE = "krasivo-jev-vllm"
# the checkout this package is installed from (editable, as `uv sync` installs it)
REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "docker" / "vllm.Dockerfile"

app = typer.Typer()


def docker(*args: str):
    subprocess.run(["docker", *args], check=True)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def serve(
    ctx: typer.Context,
    model: Annotated[
        str, typer.Argument(help="folder written by `jev-model export`, or a Hub repo id pushed from one")
    ],
    port: Annotated[int, typer.Option(help="host port for http://localhost:PORT/pooling")] = 8080,
    name: Annotated[str, typer.Option(help="container name; a running one with this name is replaced")] = "jev-pointer",
    gpus: Annotated[str, typer.Option(help="`docker run --gpus` value")] = "all",
    build: Annotated[
        bool, typer.Option(help="rebuild the image first; a no-op when the sources haven't changed")
    ] = True,
    follow: Annotated[bool, typer.Option(help="follow the container's logs after it starts")] = False,
):
    """Serve an export or its Hub repo with the `jev_vllm` plugins. Arguments after the options go to vLLM (`-- --max-num-seqs 512`)."""
    if build:
        if not DOCKERFILE.exists():
            raise typer.BadParameter(f"{DOCKERFILE} not found; run from a checkout installed with `uv sync`")
        # the layer cache makes this cheap when nothing changed, and picks up edits to `src/` when something did
        docker("build", "-t", IMAGE, "-f", str(DOCKERFILE), str(REPO))
    if Path(model).is_dir():
        mounts = ["-v", f"{Path(model).resolve()}:/model:ro"]
        model = "/model"
        vllm_args = []
    else:
        # vLLM downloads the repo itself; the shared cache keeps the download across containers
        mounts = ["-v", f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface"]
        if os.environ.get("HF_TOKEN"):
            mounts += ["-e", "HF_TOKEN"]
        # the stage is for JevHolder; vLLM would download its safetensors without loading them
        vllm_args = ["--ignore-patterns", "adapter/*"]
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    docker(
        "run", "-d", "--name", name, "--gpus", gpus, "-p", f"{port}:8080", "--ipc=host",
        *mounts,
        IMAGE,
        "--model", model, "--served-model-name", "pointer", "--runner", "pooling",
        "--port", "8080", "--host", "0.0.0.0", "--gpu-memory-utilization", "0.95",
        *vllm_args,
        *ctx.args,
    )  # fmt: skip
    typer.echo(f"started {name}: http://localhost:{port}/pooling (ready when /health returns 200)")
    if follow:
        docker("logs", "-f", name)
    else:
        typer.echo(f"logs: docker logs -f {name}   stop: docker rm -f {name}")


def main():
    app()
