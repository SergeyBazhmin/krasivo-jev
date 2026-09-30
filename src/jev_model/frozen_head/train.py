import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from jev_model import losses
from jev_model.frozen_head.cache import load_partition
from jev_model.frozen_head.config import TrainConfig
from jev_model.frozen_head.evaluate import calibrate, evaluate
from jev_model.frozen_head.head import OptionHead
from jev_model.losses import lr_factor
from jev_model.metrics import soft_cross_entropy, table


class Pool:
    """Every cached training sample in one set of tensors, drawn with per-dataset mixing weights."""

    def __init__(self, data: dict[str, dict[str, torch.Tensor]], alpha: float):
        self.names = list(data)
        self.features = torch.cat([t["features"] for t in data.values()])
        self.labels = torch.cat([t["labels"] for t in data.values()])
        self.num_options = torch.cat([t["num_options"] for t in data.values()])
        sizes = torch.tensor([len(t["labels"]) for t in data.values()], dtype=torch.float64)
        self.weights = torch.cat([torch.full((int(n),), n ** alpha / n) for n in sizes])

    def batches(self, batch_size: int, generator: torch.Generator):
        while True:
            # drawn in blocks: multinomial over ~1M rows is too slow to call per step
            for idx in torch.multinomial(self.weights, batch_size * 1_000, replacement=True, generator=generator).split(batch_size):
                yield self.features[idx], self.labels[idx], self.num_options[idx]


def ce_loss(head: OptionHead, features, labels, num_options, step: int, config: TrainConfig) -> torch.Tensor:
    return soft_cross_entropy(head(features, num_options), labels, num_options).mean()


def rl_loss(head: OptionHead, features, labels, num_options, step: int, config: TrainConfig) -> torch.Tensor:
    return losses.rl_loss(head(features, num_options), labels, num_options, step, config)


LossFn = Callable[[OptionHead, torch.Tensor, torch.Tensor, torch.Tensor, int, TrainConfig], torch.Tensor]


def train(
    cache_root: Path, out_dir: Path, stage: str, config: TrainConfig, device: str,
    names: list[str], init: Path | None = None,
):
    """Trains the head on the cached features and saves the best checkpoint by macro validation NLL,
    with its temperature fitted on validation."""
    cache_config = json.loads((cache_root / "config.json").read_text())
    train_data = load_partition(cache_root, names, "train")
    validation = load_partition(cache_root, names, "validation")
    logger.info(f"train on {len(train_data)} datasets, {sum(len(t['labels']) for t in train_data.values())} prompts")

    if init is not None:
        checkpoint = torch.load(init)
        head_config = checkpoint["head_config"]
        head = OptionHead(**head_config)
        head.load_state_dict(checkpoint["head"])
        head.temperature.fill_(1.0)
    else:
        head_config = {"hidden_size": cache_config["hidden_size"], "width": config.width, "depth": config.depth,
                       "dropout": config.dropout}
        head = OptionHead(**head_config)
    head.to(device)
    loss_fn: LossFn = {"ce": ce_loss, "rl": rl_loss}[stage]

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config))
    batches = Pool(train_data, config.alpha).batches(config.batch_size, generator)

    out_dir.mkdir(parents=True, exist_ok=True)
    best, history, running = math.inf, [], 0.0
    progress = tqdm(range(config.steps), desc=stage)
    for step in progress:
        head.train()
        features, labels, num_options = (t.to(device, non_blocking=True) for t in next(batches))
        loss = loss_fn(head, features, labels, num_options, step, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        value = loss.item()
        running += value
        progress.set_postfix(loss=f"{value:.4f}", refresh=False)
        if (step + 1) % config.eval_every == 0 or step + 1 == config.steps:
            macro = evaluate(head, validation, device)["macro"]
            history.append({"step": step + 1, "loss": running / config.eval_every} | macro)
            running = 0.0
            logger.info(json.dumps(history[-1]))
            if macro["nll"] < best:
                best = macro["nll"]
                torch.save({"head": head.state_dict(), "head_config": head_config}, out_dir / "head.pt")

    checkpoint = torch.load(out_dir / "head.pt")
    head.load_state_dict(checkpoint["head"])
    temperature = calibrate(head, validation, device)
    metrics = evaluate(head, validation, device)
    logger.info(f"temperature {temperature:.3f}, calibrated validation:\n{table(metrics)}")
    torch.save(
        {"head": head.state_dict(), "head_config": head_config, "cache": cache_config,
         "cache_root": str(cache_root.resolve()), "stage": stage,
         "train": asdict(config), "datasets": names},
        out_dir / "head.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "validation.json").write_text(json.dumps(metrics, indent=2))
