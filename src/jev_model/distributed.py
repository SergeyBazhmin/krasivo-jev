"""Several training processes under `accelerate launch`, one per GPU. The launcher sets RANK and WORLD_SIZE, so
these helpers read them without importing torch; a plain run is one process and every helper is a no-op."""

import os
import sys
from datetime import timedelta
from pathlib import Path

from loguru import logger

# how long a process may wait for the others, e.g. while the main one saves a checkpoint
TIMEOUT = timedelta(hours=1)


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup(run_dir: Path, cpu: bool = False) -> Path:
    """Starts the process group, keeps only warnings in the logs of all but the main process, and returns the main
    process's `run_dir`: each one picks its own timestamp, and they can differ by a second."""
    if world_size() == 1:
        return run_dir
    from accelerate import PartialState
    from accelerate.utils import broadcast_object_list

    # accelerate only groups CPU processes when asked to
    state = PartialState(cpu=cpu, timeout=TIMEOUT)
    if state.num_processes != world_size():
        raise RuntimeError(f"WORLD_SIZE is {world_size()} but accelerate started {state.num_processes} process(es)")
    if not is_main():
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    return broadcast_object_list([run_dir])[0]


def barrier():
    if world_size() > 1:
        from accelerate import PartialState

        PartialState().wait_for_everyone()
