"""Check the playground HTTP contract; optionally load a real saved run."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from jev_model import models
from jev_model.config import dump
from jev_model.pointer.config import PointerConfig
from jev_model.ui.app import create_app

SAMPLE = {
    "state": "The answer is blue.",
    "question": "Which color?",
    "options": [{"id": "red", "text": "red"}, {"id": "blue", "text": "blue"}],
}


class FakePredictor:
    def predict(self, questions):
        if questions[0][1] == "too long":
            raise ValueError("question exceeds the context window")
        return [[0.25, 0.75] for _ in questions]


def check_contract(root: Path):
    from datasets import Dataset, DatasetDict

    data_dir = root / "data"
    row = {**SAMPLE, "label": [0.0, 1.0], "type": "choice"}
    DatasetDict({"test": Dataset.from_list([row])}).save_to_disk(str(data_dir / "example"))
    run = root / "runs" / "pointer" / "example"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"model": "pointer", "stages": ["ce"]}))
    (run / "config.json").write_text(dump(PointerConfig(data_dir=data_dir)))
    with (
        patch.object(models["pointer"], "predictor", return_value=FakePredictor()) as factory,
        TestClient(create_app(root / "runs"), raise_server_exceptions=False) as client,
    ):
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/api/model").json() == {"loaded": None}
        assert client.get("/api/runs").json()["runs"][0]["run"] == "pointer/example"
        assert client.post("/api/predict", json=SAMPLE).status_code == 409
        request = {"run": "pointer/example", "device": "cpu"}
        response = client.post("/api/model", json=request)
        assert response.status_code == 200, response.text
        model_id = response.json()["loaded"]["id"]
        assert client.post("/api/model", json=request).json()["loaded"]["id"] == model_id
        assert factory.call_count == 1
        # Invalid selections keep the currently loaded model available.
        assert client.post("/api/model", json={**request, "stage": "rl"}).status_code == 400
        assert client.post("/api/model", json={**request, "run": "../outside"}).status_code == 400
        assert client.get("/api/model").json()["loaded"]["id"] == model_id
        result = client.post("/api/predict", json={**SAMPLE, "model_id": model_id})
        assert result.status_code == 200, result.text
        assert result.json()["prediction"] == {"id": "blue", "text": "blue", "probability": 0.75}
        assert len(result.json()["options"]) == 2
        assert result.json()["elapsed_ms"] >= 0
        assert client.post("/api/predict", json={**SAMPLE, "options": ["red", "blue"]}).status_code == 200
        assert client.post("/api/predict", json={**SAMPLE, "model_id": "stale"}).status_code == 409
        for invalid in [
            {"question": " "},
            {"options": ["red"]},
            {"options": ["red", " "]},
            {"options": [{"id": "same", "text": "red"}, {"id": "same", "text": "blue"}]},
            {"question": "too long"},
        ]:
            assert client.post("/api/predict", json={**SAMPLE, **invalid}).status_code == 422
        assert client.get("/api/datasets").json()["datasets"] == ["example"]
        sample = client.get("/api/sample", params={"dataset": "example"})
        assert sample.status_code == 200, sample.text
        assert sample.json()["sample"]["label"] == [0.0, 1.0]
        assert sample.json()["count"] == 1
        assert client.get("/api/sample", params={"dataset": "example", "index": 1}).status_code == 404
        assert client.get("/api/sample", params={"dataset": "../outside"}).status_code == 404
        assert client.delete("/api/model").json() == {"loaded": None}
        assert client.post("/api/predict", json=SAMPLE).status_code == 409
    print("UI API: model lifecycle, validation, prediction, datasets and static assets OK", flush=True)


def check_real_run(run: Path, device: str):
    run = run.expanduser().resolve()
    with TestClient(create_app(run.parent), raise_server_exceptions=False) as client:
        response = client.post("/api/model", json={"run": run.name, "device": device})
        assert response.status_code == 200, response.text
        result = client.post("/api/predict", json=SAMPLE)
        assert result.status_code == 200, result.text
        probs = [option["probability"] for option in result.json()["options"]]
        assert len(probs) == 2 and abs(sum(probs) - 1) < 1e-4
        assert client.delete("/api/model").status_code == 200
        print(f"Real checkpoint: load, inference and unload OK ({device}); probabilities={probs}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="rujev-ui-") as scratch:
        check_contract(Path(scratch))
    if args.run_dir:
        check_real_run(args.run_dir, args.device)
