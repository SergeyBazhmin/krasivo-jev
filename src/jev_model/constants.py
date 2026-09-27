from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = REPO_DIR / "data"
# per-model artefacts that runs can share, e.g. cached backbone features: CACHE_DIR/<model>/...
CACHE_DIR = REPO_DIR / "cache"
# one directory per training run: RUNS_DIR/<model>/<run>
RUNS_DIR = REPO_DIR / "runs"

PARTITIONS = ("train", "validation", "test")
