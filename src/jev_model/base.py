import json
from pathlib import Path
from typing import Protocol

from loguru import logger

from jev_model import distributed
from jev_model.config import dump

RUN_FILE = "run.json"
CONFIG_FILE = "config.json"


class Predictor(Protocol):
    def predict(self, questions: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        """Probabilities over the options of each (state, question, options), in the given order."""
        ...


class JevModel[C]:
    """One model family and its training pipeline. The pipeline is an ordered list of `stages`;
    each stage reads what earlier ones left in the run directory and writes its own outputs there.

    Subclasses set `config_class` (a dataclass with defaults for every field) and `stages`, and
    implement `run_stage`, `evaluate` and `predictor`. Heavy imports (torch, transformers) belong
    inside those methods, so `jev-model list` works without every model's dependencies."""

    config_class: type[C]
    stages: tuple[str, ...]
    # what `train` runs when no --stage is given; later stages stay opt-in
    default_stages: tuple[str, ...] = ()
    # stages `evaluate` can score; empty for every stage
    eval_stages: tuple[str, ...] = ()
    # whether `run_stage` can train in several processes under `accelerate launch` (DDP)
    multi_gpu: bool = False

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run_stage(self, stage: str, config: C, run_dir: Path, device: str):
        raise NotImplementedError

    def evaluate(
        self, config: C, run_dir: Path, partition: str, device: str, stage: str | None = None
    ) -> dict[str, dict[str, float]]:
        """{dataset: metrics}, plus `macro`. `stage` picks which stage's weights; the last trained by default."""
        raise NotImplementedError

    def predictor(self, config: C, run_dir: Path, device: str, stage: str | None = None) -> Predictor:
        raise NotImplementedError

    def train(self, config: C, run_dir: Path, device: str, stages: list[str] | None = None):
        stages = stages or list(self.default_stages or self.stages)
        if unknown := [s for s in stages if s not in self.stages]:
            raise ValueError(f"{self.name} has no stages {unknown}; it has {list(self.stages)}")
        if distributed.world_size() > 1 and not self.multi_gpu:
            raise ValueError(f"{self.name} trains in one process; run it without `accelerate launch`")
        # always in pipeline order, whatever order they were asked in
        stages = [s for s in self.stages if s in stages]
        # under `accelerate launch` every process runs this; the main one writes the run files, and only after
        # every process has read the old ones
        distributed.barrier()
        if distributed.is_main():
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / CONFIG_FILE).write_text(dump(config))
        for stage in stages:
            logger.info(f"[{self.name}] stage {stage} -> {run_dir}")
            self.run_stage(stage, config, run_dir, device)
            # the next stage reads what the main process saved
            distributed.barrier()
            if distributed.is_main():
                done = {*read_run(run_dir).get("stages", []), stage}
                write_run(run_dir, {"model": self.name, "stages": [s for s in self.stages if s in done]})


def read_run(run_dir: Path) -> dict:
    """{"model": name, "stages": [finished stages]}, or {} for a directory no model has written."""
    path = run_dir / RUN_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def write_run(run_dir: Path, info: dict):
    (run_dir / RUN_FILE).write_text(json.dumps(info, indent=2))
