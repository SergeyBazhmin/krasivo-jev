"""Exercise both training pipelines with a tiny local Qwen and state-answerable data, without downloads."""

import argparse
import math
from datetime import datetime
from pathlib import Path


def fixtures(root: Path) -> tuple[Path, Path]:
    import torch
    from datasets import Dataset, DatasetDict
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample
    from jev_model.pointer.encode import SPECIAL

    torch.manual_seed(0)
    model_dir = root / "backbone"
    words = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        *SPECIAL,
        "The",
        "answer",
        "is",
        "red",
        "blue",
        "green",
        "Which",
        "color?",
        "Example",
        "None",
        "of",
        "the",
        "above",
        *map(str, range(64)),
    ]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=SPECIAL,
    )
    tokenizer.save_pretrained(model_dir)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=vocab["[PAD]"],
            eos_token_id=vocab["[EOS]"],
            tie_word_embeddings=True,
        )
    ).save_pretrained(model_dir)
    data_dir = root / "data"
    parts = {}
    offset = 0
    for part, count in (("train", 32), ("validation", 8), ("test", 8)):
        samples = []
        for i in range(offset, offset + count):
            options = ["red", "blue", "green"][: 2 + i % 2]
            answer = options[i % len(options)]
            samples.append(
                make_sample(f"The answer is {answer} Example {i}", "Which color?", make_options(options), answer)
            )
        parts[part] = Dataset.from_list(samples, features=SAMPLE_FEATURES).add_column("type", ["choice"] * count)
        offset += count
    DatasetDict(parts).save_to_disk(data_dir / "smoke")
    return model_dir, data_dir


def check_metrics(metrics: dict, count: int):
    assert metrics["smoke"]["n"] == count, metrics
    assert all(math.isfinite(value) for value in metrics["macro"].values()), metrics


def check_calibration(run_dir: Path, stage: str):
    import json

    history = json.loads((run_dir / stage / "history.json").read_text())
    validation = json.loads((run_dir / stage / "validation.json").read_text())
    # frozen_head uses the same validation rows for checkpoint selection and calibration.
    if run_dir.name == "frozen_head":
        assert validation["macro"]["nll"] <= min(row["nll"] for row in history) + 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="", help="auto: CUDA, MPS, CPU; use cpu to force CPU")
    parser.add_argument("--steps", type=int, default=4, help="optimizer steps per CE/RL stage")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/smoke") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    from jev_model import models
    from jev_model.cli import default_device
    from jev_model.frozen_head.config import CacheConfig, FrozenHeadConfig
    from jev_model.frozen_head.config import TrainConfig as HeadTrainConfig
    from jev_model.pointer.config import PointerConfig
    from jev_model.pointer.config import TrainConfig as PointerTrainConfig

    device = default_device(args.device)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_dir, data_dir = fixtures(root)
    configs = {
        "frozen_head": FrozenHeadConfig(
            datasets=["smoke"],
            data_dir=data_dir,
            cache_dir=root / "cache",
            batch_tokens=128,
            cache=CacheConfig(model=str(model_dir), max_length=128, views=2, max_samples=6),
            ce=HeadTrainConfig(steps=args.steps, batch_size=2, width=16, warmup=1, eval_every=2),
            rl=HeadTrainConfig(steps=args.steps, batch_size=2, warmup=1, eval_every=2, group=4),
        ),
        "pointer": PointerConfig(
            datasets=["smoke"],
            data_dir=data_dir,
            max_samples=6,
            model=str(model_dir),
            lora=2,
            lora_targets="qv",
            head_dim=8,
            max_length=128,
            eval_batch_tokens=128,
            gradient_checkpointing=True,
            special_embeddings=True,
            ce=PointerTrainConfig(
                steps=args.steps, sampling="sized", batch_tokens=128, accum=1, warmup=1, eval_every=2, eval_samples=4
            ),
            rl=PointerTrainConfig(
                steps=args.steps, batch_tokens=128, accum=1, warmup=1, eval_every=2, eval_samples=4, group=4
            ),
        ),
    }
    for name, config in configs.items():
        model = models[name]
        run_dir = root / name
        print(f"{name}: CE + RL on {device}", flush=True)
        model.train(config, run_dir, device, list(model.stages))
        for stage in ("ce", "rl"):
            check_calibration(run_dir, stage)
            check_metrics(model.evaluate(config, run_dir, "test", device, stage), 6)
            predictor = model.predictor(config, run_dir, device, stage)
            (probs,) = predictor.predict([("The answer is blue", "Which color?", ["red", "blue", "green"])])
            assert len(probs) == 3 and all(math.isfinite(p) and 0 <= p <= 1 for p in probs), probs
            assert abs(sum(probs) - 1) < 1e-5, probs
            del predictor
        print(f"{name}: train, calibration, checkpoint reload, eval and predict OK", flush=True)
    print(f"artifacts: {root}")


if __name__ == "__main__":
    main()
