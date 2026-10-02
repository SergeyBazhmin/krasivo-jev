from dataclasses import dataclass, field
from pathlib import Path

from jev_model.constants import DATA_DIR


@dataclass
class TrainConfig:
    # optimizer steps; each one accumulates `accum` batches of at most `batch_tokens` padded tokens
    steps: int = 5_000
    batch_tokens: int = 16_384
    # sized sampling only: samples per batch, whatever their length, in place of `batch_tokens`
    batch_size: int = 0
    accum: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup: int = 200
    eval_every: int = 500
    # validation rows per dataset scored during training; the final metrics use the whole partition
    eval_samples: int = 200
    # how training samples are drawn: weighted (per-dataset weights, with replacement) or sized (each pass
    # shuffles the whole training set and cuts it into batches of `batch_size` samples in that order)
    sampling: str = "weighted"
    # sized sampling only: train for this many passes over the training set instead of `steps`; 0 keeps `steps`
    epochs: int = 0
    # weighted sampling only: each dataset is drawn in proportion to rows ** alpha (1 = by size, 0 = uniform)
    alpha: float = 0.5
    # chance that a training draw shows "None of the above" in place of the true option
    none_prob: float = 0.1
    seed: int = 0
    # RL stage only
    group: int = 8
    sigma_start: float = 1.0
    sigma_end: float = 0.3
    spherical_weight: float = 0.5
    ce_weight: float = 0.0


@dataclass
class PointerConfig:
    # built datasets to use; empty for every one under data_dir
    datasets: list[str] = field(default_factory=list)
    data_dir: Path = DATA_DIR
    model: str = "unsloth/Qwen3.5-2B-Base"
    # LoRA rank, and which projections get an adapter: all, attn or qv
    lora: int = 16
    lora_targets: str = "all"
    # width of the pointer head
    head_dim: int = 256
    # also train the embedding rows of the delimiter tokens
    special_embeddings: bool = False
    # sequence tokens; the state is cut in the middle past this
    max_length: int = 2048
    # padded tokens per forward pass in evaluation; speed only
    eval_batch_tokens: int = 32_768
    ce: TrainConfig = field(default_factory=TrainConfig)
    rl: TrainConfig = field(default_factory=lambda: TrainConfig(steps=1_000, lr=2e-5))
