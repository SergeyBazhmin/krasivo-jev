from dataclasses import dataclass, field
from pathlib import Path

from jev_model.constants import DATA_DIR


@dataclass
class ZeroShotConfig:
    # built datasets to score; empty for every one under data_dir
    datasets: list[str] = field(default_factory=list)
    data_dir: Path = DATA_DIR
    # OpenAI-compatible server, such as a local vLLM
    url: str = "http://localhost:8080/v1"
    # served model name; empty for the first one the server lists
    model: str = ""
    # Hub tokenizer that measures prompts for the state cut; empty for the served model name
    tokenizer: str = ""
    # prompt tokens; the state is cut in the middle past this. Keep it under the server's max_model_len
    max_length: int = 4096
    # rows per dataset and partition, chosen by a hash of the content; 0 for all
    max_samples: int = 0
    # requests in flight
    concurrency: int = 16
    retries: int = 5
    # let a reasoning model think before it answers
    thinking: bool = False
    # answer budget; the answer is an option number
    max_new_tokens: int = 16
    # the budget in place of max_new_tokens when thinking
    think_tokens: int = 2048
