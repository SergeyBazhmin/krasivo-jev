from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve().absolute() / "data"

# per split; larger splits are subsampled keeping the label distribution
MAX_SAMPLES = 10_000
