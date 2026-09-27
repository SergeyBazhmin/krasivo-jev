import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_from_disk
from loguru import logger
from tqdm import tqdm

from jev_model.data import content_key, partition
from jev_model.frozen_head.backbone import embed, length_batches, load_backbone
from jev_model.frozen_head.config import MAX_OPTIONS, CacheConfig
from jev_model.prompt import encode

def view_order(sample: dict, view: int) -> list[int]:
    """Option order for one view: view 0 keeps the source order, others shuffle it, so no head
    slot can learn a prior for an answer."""
    order = list(range(len(sample["options"])))
    if view:
        random.Random(f"view{view}:{content_key(sample)}:{sample['question']}").shuffle(order)
    return order


def build(
    names: list[str], data_dir: Path, root: Path, config: CacheConfig, device: str, batch_tokens: int,
    overwrite: bool = False,
):
    """Runs the frozen backbone once per prompt and saves the last-token hidden states under `root`.
    Datasets already cached there are skipped."""
    if not overwrite and all((root / name / "done").exists() for name in names):
        logger.info(f"all {len(names)} datasets cached in {root}")
        return
    decoder, tokenizer = load_backbone(config.model, device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(asdict(config) | {"hidden_size": decoder.config.hidden_size}))
    for name in tqdm(names, desc="cache"):
        out = root / name
        if (out / "done").exists() and not overwrite:
            logger.info(f"[{name}] cached, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        for part, samples in partition(load_from_disk(data_dir / name)).items():
            prompts, labels, rows, views, skipped = [], [], [], [], 0
            for row, sample in enumerate(tqdm(samples, desc=f"[{name}] {part} tokenize", leave=False)):
                if not 2 <= len(sample["options"]) <= MAX_OPTIONS:
                    skipped += 1
                    continue
                orders = [view_order(sample, view) for view in range(config.views if part == "train" else 1)]
                encoded = [
                    encode(
                        tokenizer, sample["state"], sample["question"],
                        [sample["options"][i]["text"] for i in order], config.max_length,
                    )
                    for order in orders
                ]
                if any(ids is None for ids in encoded):
                    skipped += 1
                    continue
                for view, (order, ids) in enumerate(zip(orders, encoded)):
                    prompts.append(ids)
                    labels.append([sample["label"][i] for i in order])
                    rows.append(row)
                    views.append(view)
            features = torch.empty(0)
            if prompts:
                batches = list(length_batches([len(p) for p in prompts], batch_tokens))
                chunks, index = [], []
                for batch in tqdm(batches, desc=f"[{name}] {part}", leave=False):
                    chunks.append(embed(decoder, pad_id, [prompts[i] for i in batch], config.layer).half())
                    index.extend(batch)
                features = torch.empty(len(prompts), chunks[0].shape[1], dtype=torch.float16)
                features[torch.tensor(index)] = torch.cat(chunks)
            padded = torch.zeros(len(labels), MAX_OPTIONS, dtype=torch.float32)
            for i, label in enumerate(labels):
                padded[i, : len(label)] = torch.tensor(label)
            torch.save(
                {
                    "features": features,
                    "labels": padded,
                    "num_options": torch.tensor([len(label) for label in labels], dtype=torch.int16),
                    "rows": torch.tensor(rows, dtype=torch.int32),
                    "views": torch.tensor(views, dtype=torch.int8),
                },
                out / f"{part}.pt",
            )
            logger.info(f"[{name}] {part}: {len(prompts)} prompts from {len(samples)} samples, {skipped} skipped")
        (out / "done").touch()


def load_partition(root: Path, names: list[str], part: str) -> dict[str, dict[str, torch.Tensor]]:
    """{dataset: tensors} for the datasets that have rows in `part`."""
    loaded = {}
    for name in tqdm(names, desc=f"load {part}", leave=False):
        path = root / name / f"{part}.pt"
        if path.exists() and (tensors := torch.load(path))["labels"].shape[0]:
            loaded[name] = tensors
    return loaded


def cached_names(root: Path) -> list[str]:
    return sorted(p.parent.name for p in root.glob("*/done"))
