import contextlib
import itertools
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
from jev_model.pointer.data import POOLS, Draw, Item, SizedEpochPool, load_items
from jev_model.pointer.evaluate import Outputs, pooled_temperature, run, score
from jev_model.pointer.model import PointerModel


def batch_loss(
    model: PointerModel, forward: torch.nn.Module, draws: list[Draw], stage: str, step: int, config: TrainConfig
) -> torch.Tensor:
    """`forward` is `model` itself, or its DDP wrapper, which syncs the gradients."""
    logits, num_options = forward([model.encoder.assemble(draw.item.tokens, draw.order) for draw in draws])
    # the labels follow the options into the order they were shown in
    labels = torch.zeros_like(logits)
    for row, draw in enumerate(draws):
        labels[row, : len(draw.order)] = torch.tensor([draw.item.label[i] for i in draw.order])
    if stage == "rl":
        return losses.rl_loss(logits, labels, num_options, step, config)
    return soft_cross_entropy(logits, labels, num_options).mean()


def run_shared(accelerator, model: PointerModel, data: dict[str, list[Item]], batch_tokens: int) -> dict[str, Outputs]:
    """`run` with the datasets dealt out to the processes, largest first; every process gets every output."""
    from accelerate.utils import gather_object

    by_size = sorted(data, key=lambda name: -sum(item.length for item in data[name]))
    mine = {name: data[name] for name in by_size[accelerator.process_index :: accelerator.num_processes]}
    outputs = {}
    for part in gather_object([run(model, mine, batch_tokens)]):
        outputs |= part
    return {name: outputs[name] for name in data}


def train(config: PointerConfig, stage: str, out_dir: Path, device: str, names: list[str], init: Path | None = None):
    """Trains the LoRA and the head and saves the best checkpoint by macro validation NLL, with its
    temperature fitted on validation. Under `accelerate launch` each process trains a DDP replica on its own GPU
    and its own share of the batches, so one optimizer step sees `accum` batches from every process."""
    from accelerate import Accelerator

    # the forward pass autocasts to bf16 itself
    accelerator = Accelerator(mixed_precision="no", cpu=device == "cpu")
    rank, world = accelerator.process_index, accelerator.num_processes
    if world > 1:
        device = str(accelerator.device)
    stage_config: TrainConfig = getattr(config, stage)
    torch.manual_seed(stage_config.seed)
    if init is not None:
        model = pointer.load(init, device, trainable=True)
        model.head.temperature.fill_(1.0)
    else:
        model = pointer.create(config, device)
    parameters = model.trainable_parameters()
    # DDP copies rank 0's weights to the others; past here each process needs its own dropout and RL noise
    torch.manual_seed(stage_config.seed + rank)
    forward = accelerator.prepare_model(model) if world > 1 else model
    logger.info(f"{sum(p.numel() for p in parameters):,} trainable parameters")

    train_data = load_items(names, config.data_dir, "train", model.encoder)
    probe = load_items(names, config.data_dir, "validation", model.encoder, limit=stage_config.eval_samples)
    logger.info(f"train on {len(train_data)} datasets, {sum(map(len, train_data.values()))} samples")

    if stage_config.sampling not in POOLS:
        raise ValueError(f"unknown sampling {stage_config.sampling!r}, expected one of {sorted(POOLS)}")
    pool = POOLS[stage_config.sampling](
        train_data,
        stage_config.alpha,
        stage_config.seed,
        model.encoder,
        stage_config.none_prob,
        stage_config.drop_prob,
    )
    generator = torch.Generator().manual_seed(stage_config.seed)
    if stage_config.epochs:
        if not isinstance(pool, SizedEpochPool):
            raise ValueError('epochs needs sampling = "sized"')
        count = pool.count_batches(stage_config.epochs, stage_config.batch_tokens, generator)
        stage_config = replace(stage_config, steps=math.ceil(count / (stage_config.accum * world)))
        logger.info(f"{stage_config.epochs} epochs: {count} batches, {stage_config.steps} steps")
    # every process walks the same stream and keeps every `world`-th batch
    batches = itertools.islice(pool.batches(stage_config.batch_tokens, generator), rank, None, world)
    optimizer = torch.optim.AdamW(parameters, lr=stage_config.lr, weight_decay=stage_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: losses.lr_factor(step, stage_config))
    saved = {"stage": stage, "train": asdict(stage_config), "datasets": names}

    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    best, history, running, since = math.inf, [], 0.0, 0
    progress = tqdm(range(stage_config.steps), desc=stage, disable=not accelerator.is_main_process)
    for step in progress:
        forward.train()
        value = 0.0
        for micro in range(stage_config.accum):
            # gradients are averaged over the processes only on the last batch of the step
            sync = accelerator.no_sync(forward) if micro + 1 < stage_config.accum else contextlib.nullcontext()
            with sync:
                loss = batch_loss(model, forward, next(batches), stage, step, stage_config) / stage_config.accum
                accelerator.backward(loss)
            value += loss.item()
        accelerator.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running, since = running + value, since + 1
        progress.set_postfix(loss=f"{value:.4f}", refresh=False)
        if (step + 1) % stage_config.eval_every == 0 or step + 1 == stage_config.steps:
            outputs = run_shared(accelerator, model, probe, config.eval_batch_tokens)
            macro = score(outputs, model.head.temperature.item())["macro"]
            loss = accelerator.reduce(torch.tensor(running / since, device=device), "mean").item()
            history.append({"step": step + 1, "loss": loss} | macro)
            running, since = 0.0, 0
            logger.info(json.dumps(history[-1]))
            if macro["nll"] < best:
                best = macro["nll"]
                if accelerator.is_main_process:
                    model.save(out_dir, **saved)
            accelerator.wait_for_everyone()

    # the best weights on the full validation partition, for the temperature and the reported scores
    del model, forward, optimizer
    model = pointer.load(out_dir, device)
    validation = load_items(names, config.data_dir, "validation", model.encoder)
    outputs = run_shared(accelerator, model, validation, config.eval_batch_tokens)
    if not accelerator.is_main_process:
        return
    temperature = pooled_temperature(outputs)
    model.head.temperature.fill_(temperature)
    metrics = score(outputs, temperature)
    logger.info(f"temperature {temperature:.3f}, calibrated validation:\n{table(metrics)}")
    model.save(out_dir, **saved)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "validation.json").write_text(json.dumps(metrics, indent=2))
