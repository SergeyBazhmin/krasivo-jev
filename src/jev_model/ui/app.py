"""Local model playground: HTTP API and a browser UI served from the same origin."""

import gc
import math
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from jev_model import models
from jev_model.base import CONFIG_FILE, RUN_FILE, Predictor, read_run
from jev_model.config import load_config, read_file
from jev_model.constants import DATA_DIR, RUNS_DIR


class LoadRequest(BaseModel):
    run: str = Field(min_length=1)
    stage: str | None = None
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"


class Option(BaseModel):
    id: str
    text: str

    @field_validator("id", "text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("option id and text must not be blank")
        return value


class PredictRequest(BaseModel):
    state: str = ""
    question: str
    options: list[str | Option] = Field(min_length=2)
    model_id: str | None = None

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("options")
    @classmethod
    def valid_options(cls, values: list[str | Option]) -> list[Option]:
        options = [Option(id=str(i), text=value) if isinstance(value, str) else value for i, value in enumerate(values)]
        if len({option.id for option in options}) != len(options):
            raise ValueError("option ids must be unique")
        return options


class ModelSession:
    """One resident predictor; loading, unloading and inference share a lock."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir.expanduser().resolve()
        self.lock = Lock()
        self.predictor: Predictor | None = None
        self.info: dict | None = None
        self.config = None

    def run_path(self, name: str) -> Path:
        path = (self.runs_dir / name).resolve()
        if not path.is_relative_to(self.runs_dir):
            raise HTTPException(400, "Run must be inside the configured runs directory")
        return path

    def describe(self, path: Path) -> dict:
        info = read_run(path)
        if not isinstance(info, dict) or not isinstance(info.get("stages", []), list):
            raise TypeError("Invalid run.json")
        name = info.get("model")
        if name not in models or not (path / CONFIG_FILE).is_file():
            raise ValueError("Not a registered model run with config.json")
        # `embed` creates features, not a checkpoint that can answer questions.
        stages = [stage for stage in info.get("stages", []) if stage in models[name].stages and stage != "embed"]
        return {"run": path.relative_to(self.runs_dir).as_posix(), "model": name, "stages": stages}

    def runs(self) -> list[dict]:
        result = []
        paths = sorted(self.runs_dir.rglob(RUN_FILE), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                if not path.resolve().is_relative_to(self.runs_dir):
                    continue
                result.append(self.describe(path.parent))
            except (OSError, ValueError, TypeError):
                logger.warning("Skipping invalid run: {}", path.parent)
        return result

    def clear(self):
        """Called under the lock, including when replacing a model on a small-memory Mac."""
        device = self.info.get("device") if self.info else None
        self.predictor = None
        self.info = None
        self.config = None
        gc.collect()
        if device in ("cuda", "mps"):
            import torch

            if device == "cuda":
                torch.cuda.empty_cache()
            else:
                torch.mps.empty_cache()

    def load(self, request: LoadRequest) -> dict:
        from jev_model.cli import default_device

        with self.lock:
            path = self.run_path(request.run)
            try:
                info = self.describe(path)
                if not info["stages"]:
                    raise ValueError("This run has no completed prediction stage")
                stage = request.stage or info["stages"][-1]
                if stage not in info["stages"]:
                    raise ValueError(f"Choose a completed stage: {info['stages']}")
                model = models[info["model"]]
                config = load_config(model.config_class, base=read_file(path / CONFIG_FILE))
                device = default_device("" if request.device == "auto" else request.device)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise HTTPException(400, str(error)) from error
            if self.info and all(
                self.info[key] == value for key, value in {"run": info["run"], "stage": stage, "device": device}.items()
            ):
                return self.info
            self.clear()
            started = perf_counter()
            try:
                self.predictor = model.predictor(config, path, device, stage)
            except Exception as error:
                self.clear()
                logger.exception("Model loading failed")
                raise HTTPException(400, f"{type(error).__name__}: {error}") from error
            self.config = config
            self.info = {
                **info,
                "id": uuid4().hex,
                "stage": stage,
                "device": device,
                "load_ms": round((perf_counter() - started) * 1000, 1),
            }
            return self.info

    def predict(self, request: PredictRequest) -> dict:
        with self.lock:
            if self.predictor is None:
                raise HTTPException(409, "Load a model first")
            if request.model_id is not None and request.model_id != self.info["id"]:
                raise HTTPException(409, "The loaded model changed. Reload the page and try again")
            started = perf_counter()
            try:
                (probs,) = self.predictor.predict(
                    [(request.state, request.question, [option.text for option in request.options])]
                )
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            if (
                len(probs) != len(request.options)
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)
                or not math.isclose(sum(probs), 1.0, abs_tol=1e-4)
            ):
                raise RuntimeError("Predictor returned an invalid probability distribution")
            rows = [{**option.model_dump(), "probability": float(p)} for option, p in zip(request.options, probs)]
            best = max(range(len(probs)), key=probs.__getitem__)
            return {
                "model": self.info,
                "options": rows,
                "prediction": rows[best],
                "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            }

    def data_dir(self) -> Path:
        with self.lock:
            return Path(getattr(self.config, "data_dir", DATA_DIR))


@lru_cache(maxsize=4)
def dataset_partitions(data_dir: Path, name: str) -> dict[str, list[dict]]:
    from datasets import load_from_disk

    from jev_model.data import partition

    return partition(load_from_disk(str(data_dir / name)))


def create_app(runs_dir: Path = RUNS_DIR) -> FastAPI:
    session = ModelSession(runs_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        with session.lock:
            session.clear()
        dataset_partitions.cache_clear()

    app = FastAPI(title="rujev playground", lifespan=lifespan)
    app.state.session = session

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        from fastapi.responses import JSONResponse

        logger.opt(exception=error).error("Playground request failed")
        return JSONResponse(status_code=500, content={"detail": f"{type(error).__name__}: {error}"})

    @app.get("/api/runs")
    def runs():
        return {"runs_dir": str(session.runs_dir), "runs": session.runs()}

    @app.get("/api/model")
    def status():
        return {"loaded": session.info}

    @app.post("/api/model")
    def load_model(request: LoadRequest):
        return {"loaded": session.load(request)}

    @app.delete("/api/model")
    def unload_model():
        with session.lock:
            session.clear()
        return {"loaded": None}

    @app.post("/api/predict")
    def predict(request: PredictRequest):
        return session.predict(request)

    @app.get("/api/datasets")
    def datasets():
        from jev_model.data import built_names

        data_dir = session.data_dir()
        return {"datasets": built_names(data_dir)}

    @app.get("/api/sample")
    def sample(
        dataset: str, partition: Literal["train", "validation", "test"] = "test", index: int = Query(default=0, ge=0)
    ):
        from jev_model.data import built_names

        data_dir = session.data_dir()
        if dataset not in built_names(data_dir):
            raise HTTPException(404, "Unknown built dataset")
        rows = dataset_partitions(data_dir, dataset)[partition]
        if index >= len(rows):
            raise HTTPException(404, f"Row {index} is out of range; this partition has {len(rows)} rows")
        return {"sample": rows[index], "index": index, "count": len(rows)}

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    return app
