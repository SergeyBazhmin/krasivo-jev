import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from jev_model import losses
from jev_model.metrics import soft_cross_entropy, table
from jev_model.pointer import model as pointer
from jev_model.pointer.config import PointerConfig, TrainConfig
from jev_model.pointer.data import POOLS, Draw, SizedEpochPool, load_items
from jev_model.pointer.evaluate import evaluate, pooled_temperature, run, score
from jev_model.pointer.model import PointerModel


def batch_loss(model: PointerModel, draws: list[Draw], stage: str, step: int, config: TrainConfig) -> torch.Tensor:
    logits, num_options = model([model.encoder.assemble(draw.item.tokens, draw.order) for draw in draws])
    # the labels follow the options into the order they were shown in
    labels = torch.zeros_like(logits)
    for row, draw in enumerate(draws):
        labels[row, : len(draw.order)] = torch.tensor([draw.item.label[i] for i in draw.order])
    weights = torch.tensor([draw.weight for draw in draws], device=logits.device)
    if stage == "rl":
        return losses.rl_loss(logits, labels, num_options, step, config, weights)
    return (soft_cross_entropy(logits, labels, num_options) * weights).mean()


def train(config: PointerConfig, stage: str, out_dir: Path, device: str, names: list[str], init: Path | None = None):
    """Trains the LoRA and the head and saves the best checkpoint by macro validation NLL, with its
    temperature fitted on validation."""
    stage_config: TrainConfig = getattr(config, stage)
    torch.manual_seed(stage_config.seed)
    if init is not None:
        model = pointer.load(init, device, trainable=True)
        model.head.temperature.fill_(1.0)
    else:
        model = pointer.create(config, device)
    parameters = model.trainable_parameters()
    logger.info(f"{sum(p.numel() for p in parameters):,} trainable parameters")

    train_data = load_items(names, config.data_dir, "train", model.encoder)
    probe = load_items(names, config.data_dir, "validation", model.encoder, limit=stage_config.eval_samples)
    logger.info(f"train on {len(train_data)} datasets, {sum(map(len, train_data.values()))} samples")

    if stage_config.sampling not in POOLS:
        raise ValueError(f"unknown sampling {stage_config.sampling!r}, expected one of {sorted(POOLS)}")
    pool = POOLS[stage_config.sampling](
        train_data, stage_config.alpha, stage_config.seed, model.encoder, stage_config.none_prob,
        stage_config.drop_prob,
    )
    generator = torch.Generator().manual_seed(stage_config.seed)
    if stage_config.epochs:
        if not isinstance(pool, SizedEpochPool):
            raise ValueError("epochs needs sampling = \"sized\"")
        count = pool.count_batches(stage_config.epochs, stage_config.batch_tokens, generator)
        stage_config = replace(stage_config, steps=math.ceil(count / stage_config.accum))
        logger.info(f"{stage_config.epochs} epochs: {count} batches, {stage_config.steps} steps")
    batches = pool.batches(stage_config.batch_tokens, generator)
    optimizer = torch.optim.AdamW(parameters, lr=stage_config.lr, weight_decay=stage_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: losses.lr_factor(step, stage_config))
    saved = {"stage": stage, "train": asdict(stage_config), "datasets": names}

    out_dir.mkdir(parents=True, exist_ok=True)
    best, history, running, since = math.inf, [], 0.0, 0
    progress = tqdm(range(stage_config.steps), desc=stage)
    for step in progress:
        model.train()
        value = 0.0
        for _ in range(stage_config.accum):
            loss = batch_loss(model, next(batches), stage, step, stage_config) / stage_config.accum
            loss.backward()
            value += loss.item()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running, since = running + value, since + 1
        progress.set_postfix(loss=f"{value:.4f}", refresh=False)
        if (step + 1) % stage_config.eval_every == 0 or step + 1 == stage_config.steps:
            macro = evaluate(model, probe, config.eval_batch_tokens)["macro"]
            history.append({"step": step + 1, "loss": running / since} | macro)
            running, since = 0.0, 0
            logger.info(json.dumps(history[-1]))
            if macro["nll"] < best:
                best = macro["nll"]
                model.save(out_dir, **saved)

    # the best weights on the full validation partition, for the temperature and the reported scores
    del model, optimizer
    model = pointer.load(out_dir, device)
    outputs = run(model, load_items(names, config.data_dir, "validation", model.encoder), config.eval_batch_tokens)
    temperature = pooled_temperature(outputs)
    model.head.temperature.fill_(temperature)
    metrics = score(outputs, temperature)
    logger.info(f"temperature {temperature:.3f}, calibrated validation:\n{table(metrics)}")
    model.save(out_dir, **saved)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "validation.json").write_text(json.dumps(metrics, indent=2))
