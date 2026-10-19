import json
import shutil
from pathlib import Path

import torch

from jev_model.pointer.model import ADAPTER_DIR, HEAD_FILE, PointerModel, load

# what vLLM resolves the exported folder to; registered by the `jev_vllm` plugin
ARCHITECTURE = "Qwen3_5PointerModel"
# the IO processor that `jev_vllm` registers; vLLM picks it from `config.json`
IO_PROCESSOR = "jev_pointer"
# copied from the base model as they are: files saved by transformers 5 may not load in vLLM's transformers 4
BASE_FILES = ["config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]


def base_dir(model: str) -> Path:
    """A local folder with the base model's config and tokenizer files."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=BASE_FILES))


def merged_tensors(model: PointerModel) -> dict[str, torch.Tensor]:
    """The text model with the LoRA (and trained delimiter rows) merged in, as `model.*`, and the head as
    `pooler.head.*`. Weights are bf16, as the base was trained; the head stays float32."""
    merged = model.decoder.merge_and_unload()
    text = merged.model if hasattr(merged, "lm_head") else merged
    tensors = {f"model.{key}": value.to(torch.bfloat16).contiguous() for key, value in text.state_dict().items()}
    for key, value in model.head.state_dict().items():
        if key != "temperature":
            tensors[f"pooler.head.{key}"] = value.float().contiguous()
    return tensors


def export(checkpoint_dir: Path, out_dir: Path, device: str):
    """A folder `vllm serve` loads with the `jev_vllm` plugin installed: merged weights, the base config with the
    pointer settings added, and the base tokenizer. The stage's `adapter/` and `head.pt` go beside them, so the same
    folder (or a Hub repo pushed from it) also loads in `JevHolder`."""
    from safetensors.torch import save_file

    model = load(checkpoint_dir, device)
    settings = model.settings
    out_dir.mkdir(parents=True, exist_ok=True)
    source = base_dir(settings["model"])
    for pattern in BASE_FILES:
        for path in source.glob(pattern):
            shutil.copyfile(path, out_dir / path.name)
    encoder = model.encoder
    config = json.loads((source / "config.json").read_text())
    config |= {
        "architectures": [ARCHITECTURE],
        "io_processor_plugin": IO_PROCESSOR,
        "pointer_head_dim": settings["head_dim"],
        "pointer_max_length": settings["max_length"],
        "pointer_close_id": encoder.close_id,
        "pointer_temperature": model.head.temperature.item(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    save_file(merged_tensors(model), out_dir / "model.safetensors", metadata={"format": "pt"})
    # vLLM only reads top-level weights, so the stage can sit in the same folder
    shutil.copytree(checkpoint_dir / ADAPTER_DIR, out_dir / ADAPTER_DIR, dirs_exist_ok=True)
    shutil.copyfile(checkpoint_dir / HEAD_FILE, out_dir / HEAD_FILE)
