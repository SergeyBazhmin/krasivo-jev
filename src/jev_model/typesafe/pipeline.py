from pathlib import Path

from jev_model.base import JevModel
from jev_model.data import resolve_names
from jev_model.typesafe.config import TypesafeConfig


class TypesafeJevModel(JevModel[TypesafeConfig]):
    """TypeSafe's hosted `jev` model behind its `system_one` API, as an external reference: each sample is one
    `noul` or `choice` question, and the answer's probabilities are the prediction. `device` is unused. Nothing is
    trained, so `train` only records the config and the scoring happens in `eval`."""

    config_class = TypesafeConfig
    stages = ("api",)
    default_stages = ("api",)

    def run_stage(self, stage: str, config: TypesafeConfig, run_dir: Path, device: str):
        (run_dir / stage).mkdir(parents=True, exist_ok=True)

    def evaluate(self, config, run_dir, partition, device, stage=None):
        from jev_model.typesafe.evaluate import evaluate

        names = resolve_names(config.datasets, config.data_dir)
        return evaluate(config, names, partition, run_dir / "api")

    def predictor(self, config, run_dir, device, stage=None):
        from jev_model.typesafe.predict import TypesafePredictor

        return TypesafePredictor(config)


typesafe_model = TypesafeJevModel(name="typesafe", description="TypeSafe jev API (system_one): option probabilities")
