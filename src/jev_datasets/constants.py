from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve().absolute()

# per split; larger splits are subsampled keeping the label distribution
MAX_SAMPLES = 10_000
