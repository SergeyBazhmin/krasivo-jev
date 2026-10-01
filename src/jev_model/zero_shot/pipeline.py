from pathlib import Path

from jev_model.base import JevModel
from jev_model.data import resolve_names
from jev_model.zero_shot.config import ZeroShotConfig


class ZeroShotJevModel(JevModel[ZeroShotConfig]):
    """A pretrained chat LM as it is, for a baseline: an OpenAI-compatible server (a local vLLM) generates an
    answer, and the option number it replies with is the prediction. `device` is unused. Nothing is trained, so
    `train` only records the config and the scoring happens in `eval`."""

    config_class = ZeroShotConfig
    stages = ("generate",)
    default_stages = ("generate",)

    def run_stage(self, stage: str, config: ZeroShotConfig, run_dir: Path, device: str):
        (run_dir / stage).mkdir(parents=True, exist_ok=True)

    def evaluate(self, config, run_dir, partition, device, stage=None):
        from jev_model.zero_shot.evaluate import evaluate

        names = resolve_names(config.datasets, config.data_dir)
        return evaluate(config, names, partition, run_dir / "generate")

    def predictor(self, config, run_dir, device, stage=None):
        from jev_model.zero_shot.predict import ZeroShotPredictor

        return ZeroShotPredictor(config)


zero_shot_model = ZeroShotJevModel(name="zero_shot", description="pretrained LM on a vLLM server: a generated answer")
