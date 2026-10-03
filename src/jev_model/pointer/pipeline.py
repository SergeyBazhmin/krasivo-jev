from pathlib import Path

from jev_model.base import JevModel, read_run
from jev_model.data import resolve_names
from jev_model.pointer.config import PointerConfig


class PointerJevModel(JevModel[PointerConfig]):
    """A causal decoder with a LoRA reads `<state> .. <q> .. <opt> .. </opt> .. <decide>`; a pointer
    head scores each option from the hidden states at its closing token and at `<decide>`.
    Stages: `ce` trains the LoRA and the head with soft cross-entropy on shuffled option orders,
    `rl` fine-tunes both with a proper-scoring-rule policy gradient starting from the `ce` weights.
    Each stage writes `RUN/<stage>/adapter` and `RUN/<stage>/head.pt`."""

    config_class = PointerConfig
    stages = ("ce", "rl")
    default_stages = ("ce",)
    multi_gpu = True

    def run_stage(self, stage: str, config: PointerConfig, run_dir: Path, device: str):
        from jev_model.pointer.train import train

        init = None
        if stage == "rl":
            init = run_dir / "ce"
            if not (init / "head.pt").exists():
                raise FileNotFoundError(f"the rl stage starts from the CE weights; run the ce stage first ({init})")
        train(config, stage, run_dir / stage, device, resolve_names(config.datasets, config.data_dir), init)

    def checkpoint(self, run_dir: Path, stage: str | None) -> Path:
        """The directory of `stage`, or of the last stage this run finished."""
        if stage is None:
            done = read_run(run_dir).get("stages", [])
            if not done:
                raise FileNotFoundError(f"no trained model in {run_dir}")
            stage = done[-1]
        elif stage not in self.stages:
            raise ValueError(f"stage must be one of {self.stages}, got {stage!r}")
        return run_dir / stage

    def evaluate(self, config, run_dir, partition, device, stage=None):
        from jev_model.pointer.data import load_items
        from jev_model.pointer.evaluate import evaluate
        from jev_model.pointer.model import load, read_checkpoint

        checkpoint = self.checkpoint(run_dir, stage)
        model = load(checkpoint, device)
        data = load_items(read_checkpoint(checkpoint)["datasets"], config.data_dir, partition, model.encoder)
        return evaluate(model, data, config.eval_batch_tokens)

    def predictor(self, config, run_dir, device, stage=None):
        from jev_model.pointer.predict import PointerPredictor

        return PointerPredictor(self.checkpoint(run_dir, stage), device)


pointer_model = PointerJevModel(name="pointer", description="LoRA Qwen decoder + pointer head over the option tokens")
