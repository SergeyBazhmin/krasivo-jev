from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve().absolute() / "data"

# per split; larger splits are subsampled keeping the label distribution
MAX_SAMPLES = 10_000

# share of train carved into each missing held-out split
HOLDOUT_PERCENT = 5
# share of all rows put in each held-out split when the source has no train split at all
EVAL_ONLY_HOLDOUT_PERCENT = 10
