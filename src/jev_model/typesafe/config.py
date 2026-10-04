from dataclasses import dataclass, field
from pathlib import Path

from jev_model.constants import DATA_DIR


@dataclass
class TypesafeConfig:
    # built datasets to score; empty for every one under data_dir
    datasets: list[str] = field(default_factory=list)
    data_dir: Path = DATA_DIR
    # API root; the key is read from TYPESAFE_API_KEY (the environment or `.env`), so it never lands in config.json
    base_url: str = "https://routerai.ru/api"
    # a pinned version, so runs stay comparable; `~typesafe/jev-latest` follows new releases
    model: str = "typesafe/jev-1.13"
    # rows per dataset and partition, chosen by a hash of the content; 0 for all
    max_samples: int = 0
    # requests in flight
    concurrency: int = 16
    # retries per request after the first attempt, on timeouts, 429 and 5xx
    retries: int = 5
    # seconds per HTTP operation
    timeout: float = 60.0
    # ask yes/no option pairs as a `noul` question (the API's yes/no primitive) rather than a two-way `choice`
    noul: bool = True
