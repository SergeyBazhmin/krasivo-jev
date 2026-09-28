from dataclasses import dataclass, field
from pathlib import Path

from jev_model.constants import CACHE_DIR, DATA_DIR

# width of the head; a sample with more options than this is skipped
MAX_OPTIONS = 255


@dataclass
class CacheConfig:
    """What the cached features depend on; runs with the same one share a cache."""

    model: str = "Qwen/Qwen3.5-2B"
    # hidden layer to read; -1 is the final, normed one
    layer: int = -1
    # prompt tokens; the state is cut in the middle past this
    max_length: int = 2048
    # option orders per training sample (validation/test get 1)
    views: int = 4

    def slug(self) -> str:
        return f"{self.model.replace('/', '--')}/L{self.layer}_T{self.max_length}_V{self.views}"


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: int = 500
    eval_every: int = 1_000
    # dataset mixing: each dataset is drawn in proportion to rows ** alpha (1 = by size, 0 = uniform)
    alpha: float = 0.5
    seed: int = 0
    # head shape, used by the CE stage; the RL stage keeps the one it starts from
    width: int = 2048
    depth: int = 2
    dropout: float = 0.1
    # RL stage only
    group: int = 8
    sigma_start: float = 1.0
    sigma_end: float = 0.3
    spherical_weight: float = 0.5
    ce_weight: float = 0.0


@dataclass
class FrozenHeadConfig:
    # built datasets to use; empty for every one under data_dir
    datasets: list[str] = field(default_factory=list)
    data_dir: Path = DATA_DIR
    cache_dir: Path = CACHE_DIR / "frozen_head"
    # padded tokens per backbone forward pass; speed only, not part of the cache key
    batch_tokens: int = 32_768
    cache: CacheConfig = field(default_factory=CacheConfig)
    ce: TrainConfig = field(default_factory=TrainConfig)
    rl: TrainConfig = field(default_factory=lambda: TrainConfig(lr=1e-4))
