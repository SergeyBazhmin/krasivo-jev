import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from jev_model.frozen_head.cache import load_partition
from jev_model.frozen_head.config import TrainConfig
from jev_model.frozen_head.evaluate import calibrate, evaluate
from jev_model.frozen_head.head import OptionHead
from jev_model.metrics import option_mask, soft_cross_entropy


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


def lr_factor(step: int, config: TrainConfig) -> float:
    if step < config.warmup:
        return (step + 1) / config.warmup
    progress = (step - config.warmup) / max(1, config.steps - config.warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def ce_loss(head: OptionHead, features, labels, num_options, step: int, config: TrainConfig) -> torch.Tensor:
    return soft_cross_entropy(head(features, num_options), labels, num_options).mean()


def proper_score_reward(q: torch.Tensor, labels: torch.Tensor, spherical_weight: float) -> torch.Tensor:
    """S_log + w * S_sph. The log score is floored at log(1e-4) = -9.21 so a single miss can't
    dominate the group's advantage."""
    log_score = (labels * q.clamp_min(1e-4).log()).sum(dim=-1)
    spherical = (labels * q).sum(dim=-1) / q.norm(dim=-1).clamp_min(1e-12)
    return log_score + spherical_weight * spherical


def rl_loss(head: OptionHead, features, labels, num_options, step: int, config: TrainConfig) -> torch.Tensor:
    """Gaussian policy over the logits (GRPO-style): perturb them `group` times, score each
    perturbed distribution with a proper scoring rule, and push the logits towards the
    perturbations that scored above the group's mean."""
    sigma = config.sigma_start + (config.sigma_end - config.sigma_start) * step / max(1, config.steps - 1)
    logits = head(features, num_options)
    mask = option_mask(num_options, logits.shape[-1])
    z = logits.masked_fill(mask, 0.0)
    count = num_options.float().unsqueeze(-1)
    noise = torch.randn((config.group, *z.shape), device=z.device) * sigma
    noise = noise.masked_fill(mask, 0.0)
    # centred over the real options: a shift of every logit changes nothing after the softmax
    noise = (noise - noise.sum(dim=-1, keepdim=True) / count).masked_fill(mask, 0.0)
    sampled = z.detach() + noise
    q = sampled.masked_fill(mask, float("-inf")).softmax(dim=-1)
    reward = proper_score_reward(q, labels, config.spherical_weight)
    advantage = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)
    log_prob = -((sampled - z) ** 2).sum(dim=-1) / (2 * sigma**2)
    loss = -(advantage.detach() * log_prob).mean()
    if config.ce_weight:
        loss = loss + config.ce_weight * soft_cross_entropy(logits, labels, num_options).mean()
    return loss


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
    print(f"train on {len(train_data)} datasets, {sum(len(t['labels']) for t in train_data.values())} prompts")

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
    for step in range(config.steps):
        head.train()
        features, labels, num_options = (t.to(device, non_blocking=True) for t in next(batches))
        loss = loss_fn(head, features, labels, num_options, step, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += loss.item()
        if (step + 1) % config.eval_every == 0 or step + 1 == config.steps:
            macro = evaluate(head, validation, device)["macro"]
            history.append({"step": step + 1, "loss": running / config.eval_every} | macro)
            running = 0.0
            print(json.dumps(history[-1]))
            if macro["nll"] < best:
                best = macro["nll"]
                torch.save({"head": head.state_dict(), "head_config": head_config}, out_dir / "head.pt")

    checkpoint = torch.load(out_dir / "head.pt")
    head.load_state_dict(checkpoint["head"])
    temperature = calibrate(head, validation, device)
    metrics = evaluate(head, validation, device)
    print(f"temperature {temperature:.3f}, calibrated validation {json.dumps(metrics['macro'])}")
    torch.save(
        {"head": head.state_dict(), "head_config": head_config, "cache": cache_config,
         "cache_root": str(cache_root.resolve()), "stage": stage,
         "train": asdict(config), "datasets": names},
        out_dir / "head.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "validation.json").write_text(json.dumps(metrics, indent=2))
