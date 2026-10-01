from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve().absolute() / "data"
# `jev translate` output, next to `data/` so the Russian copies are not taken for datasets of their own
RU_DIR = ROOT_DIR.parent / "data_ru"

# per split; larger splits are subsampled keeping the label distribution
MAX_SAMPLES = 10_000
