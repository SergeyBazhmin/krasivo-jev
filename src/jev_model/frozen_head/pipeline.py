from pathlib import Path

from jev_model.base import JevModel, read_run
from jev_model.data import resolve_names
from jev_model.frozen_head.config import FrozenHeadConfig

TRAIN_STAGES = ("ce", "rl")


class FrozenHeadModel(JevModel[FrozenHeadConfig]):
    """A frozen decoder reads the prompt once; an MLP on its last-token hidden state scores the
    options. Stages: `embed` caches the hidden states (shared by every run with the same
    `cache` config), `ce` fits the head with soft cross-entropy, `rl` fine-tunes it with a
    proper-scoring-rule policy gradient starting from the `ce` head."""

    config_class = FrozenHeadConfig
    stages = ("embed", *TRAIN_STAGES)
    default_stages = ("embed", "ce")

    def cache_root(self, config: FrozenHeadConfig) -> Path:
        return config.cache_dir / config.cache.slug()

    def names(self, config: FrozenHeadConfig) -> list[str]:
        return resolve_names(config.datasets, config.data_dir)

    def run_stage(self, stage: str, config: FrozenHeadConfig, run_dir: Path, device: str):
        if stage == "embed":
            from jev_model.frozen_head.cache import build

            build(
                self.names(config), config.data_dir, self.cache_root(config), config.cache, device, config.batch_tokens
            )
            return

        from jev_model.frozen_head.train import train

        init = None
        if stage == "rl":
            init = run_dir / "ce" / "head.pt"
            if not init.exists():
                raise FileNotFoundError(f"the rl stage starts from a CE head; run the ce stage first ({init})")
        train(self.cache_root(config), run_dir / stage, stage, getattr(config, stage), device, self.names(config), init)

    def checkpoint(self, run_dir: Path, stage: str | None) -> Path:
        """head.pt of `stage`, or of the last training stage this run finished."""
        if stage is None:
            done = [s for s in read_run(run_dir).get("stages", []) if s in TRAIN_STAGES]
            if not done:
                raise FileNotFoundError(f"no trained head in {run_dir}")
            stage = done[-1]
        elif stage not in TRAIN_STAGES:
            raise ValueError(f"stage must be one of {TRAIN_STAGES}, got {stage!r}")
        return run_dir / stage / "head.pt"

    def evaluate(self, config, run_dir, partition, device, stage=None):
        import torch

        from jev_model.frozen_head.cache import load_partition
        from jev_model.frozen_head.evaluate import evaluate
        from jev_model.frozen_head.head import OptionHead

        checkpoint = torch.load(self.checkpoint(run_dir, stage), map_location="cpu")
        head = OptionHead(**checkpoint["head_config"])
        head.load_state_dict(checkpoint["head"])
        data = load_partition(self.cache_root(config), checkpoint["datasets"], partition)
        return evaluate(head.to(device), data, device)

    def predictor(self, config, run_dir, device, stage=None):
        from jev_model.frozen_head.predict import JevPredictor

        return JevPredictor(self.checkpoint(run_dir, stage), device)


frozen_head_model = FrozenHeadModel(name="frozen_head", description="frozen Qwen decoder + 255-way option head")
