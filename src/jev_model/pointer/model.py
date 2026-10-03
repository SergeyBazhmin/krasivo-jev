import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from jev_model.metrics import option_mask
from jev_model.pointer.config import PointerConfig
from jev_model.pointer.encode import SPECIAL, Encoder

ADAPTER_DIR = "adapter"
HEAD_FILE = "head.pt"

ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
# the projections of Qwen3.5's linear-attention (Gated DeltaNet) layers
LINEAR_ATTENTION = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
# what the checkpoint needs to rebuild the model
MODEL_KEYS = ("model", "lora", "lora_targets", "head_dim", "special_embeddings", "max_length")


class PointerHead(nn.Module):
    """Scores each option by matching the hidden state at its closing token against the one at
    `<decide>`, so the head has no fixed number of slots and no slot can hold a prior."""

    def __init__(self, hidden_size: int, dim: int = 256):
        super().__init__()
        self.query = nn.Linear(hidden_size, dim)
        self.key = nn.Linear(hidden_size, dim)
        self.scale = 1 / math.sqrt(dim)
        # fitted on validation after training; divides the logits at inference
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, decide: torch.Tensor, options: torch.Tensor, num_options: torch.Tensor) -> torch.Tensor:
        """[B, d], [B, K, d] -> logits [B, K], -inf at the slots past a sample's options."""
        logits = torch.einsum("bkd,bd->bk", self.key(options), self.query(decide)) * self.scale
        return logits.masked_fill(option_mask(num_options, logits.shape[-1]), float("-inf"))


def lora_targets(kind: str, hybrid: bool) -> list[str]:
    targets = {"all": ATTENTION + MLP, "attn": ATTENTION, "qv": ["q_proj", "v_proj"]}[kind]
    return targets + LINEAR_ATTENTION if hybrid and kind != "qv" else targets


class PointerModel(nn.Module):
    """A causal decoder with a LoRA and a pointer head: one forward pass per question."""

    def __init__(self, decoder: nn.Module, tokenizer: PreTrainedTokenizerBase, head: PointerHead, settings: dict):
        super().__init__()
        self.decoder = decoder
        self.head = head
        self.settings = settings
        self.encoder = Encoder(tokenizer, settings["max_length"])
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @property
    def device(self) -> torch.device:
        return self.head.query.weight.device

    @property
    def backbone(self) -> nn.Module:
        """The text model under the LoRA, without the LM head, so no vocabulary logits are computed.
        Older adapters wrap the text model itself."""
        base = self.decoder.get_base_model()
        return base.model if hasattr(base, "lm_head") else base

    def forward(self, prompts: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits [B, K], num_options [B]) for (ids, closing-token positions) pairs from `Encoder.assemble`.
        Padding goes on the right: every layer is causal, so it cannot change a real position."""
        device = self.device
        width = max(len(ids) for ids, _ in prompts)
        slots = max(len(closes) for _, closes in prompts)
        ids = torch.full((len(prompts), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), width), dtype=torch.long)
        closes = torch.zeros((len(prompts), slots), dtype=torch.long)
        for row, (prompt, positions) in enumerate(prompts):
            ids[row, : len(prompt)] = torch.tensor(prompt)
            mask[row, : len(prompt)] = 1
            closes[row, : len(positions)] = torch.tensor(positions)
        ids, mask, closes = ids.to(device), mask.to(device), closes.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        hidden = hidden.float()
        rows = torch.arange(len(prompts), device=device)
        num_options = torch.tensor([len(positions) for _, positions in prompts], device=device)
        return self.head(hidden[rows, mask.sum(dim=1) - 1], hidden[rows.unsqueeze(-1), closes], num_options), num_options

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, out_dir: Path, **extra):
        """The adapter and the head; the base weights are loaded from `settings["model"]` again."""
        self.decoder.save_pretrained(out_dir / ADAPTER_DIR)
        torch.save({"head": self.head.state_dict(), "settings": self.settings, **extra}, out_dir / HEAD_FILE)


def load_decoder(model: str, device: str) -> nn.Module:
    """The text causal LM; its LM head is tied to the embeddings, so it costs no memory. On CUDA it is bf16, the
    dtype the forward pass autocasts to, so the frozen weights are not cast again on every batch; peft keeps the
    adapter weights in float32. Qwen3.5 checkpoints are multimodal; `AutoModelForCausalLM` loads only the
    language model."""
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device)


def create(config: PointerConfig, device: str) -> PointerModel:
    """A new model: the base decoder with a fresh LoRA and head."""
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    decoder = load_decoder(config.model, device)
    hybrid = "linear_attention" in (getattr(decoder.config, "layer_types", None) or [])
    lora = {
        "r": config.lora, "lora_alpha": 2 * config.lora, "lora_dropout": 0.05,
        "target_modules": lora_targets(config.lora_targets, hybrid),
    }
    if config.special_embeddings:
        lora["trainable_token_indices"] = {"embed_tokens": [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL]}
    decoder = get_peft_model(decoder, LoraConfig(task_type="CAUSAL_LM", **lora))
    head = PointerHead(decoder.config.hidden_size, config.head_dim).to(device)
    return PointerModel(decoder, tokenizer, head, {key: getattr(config, key) for key in MODEL_KEYS})


def load(checkpoint_dir: Path, device: str, trainable: bool = False) -> PointerModel:
    """A model saved by `PointerModel.save`."""
    from peft import PeftModel

    checkpoint = torch.load(checkpoint_dir / HEAD_FILE, map_location="cpu")
    settings = checkpoint["settings"]
    tokenizer = AutoTokenizer.from_pretrained(settings["model"])
    base = load_decoder(settings["model"], device)
    adapter = json.loads((checkpoint_dir / ADAPTER_DIR / "adapter_config.json").read_text())
    if adapter["task_type"] == "FEATURE_EXTRACTION":
        # an older adapter: it wraps the text model, not the causal LM
        base = base.model
    decoder = PeftModel.from_pretrained(base, checkpoint_dir / ADAPTER_DIR, is_trainable=trainable)
    head = PointerHead(decoder.config.hidden_size, settings["head_dim"])
    head.load_state_dict(checkpoint["head"])
    model = PointerModel(decoder, tokenizer, head.to(device), settings)
    return model if trainable else model.eval().requires_grad_(False)


def read_checkpoint(checkpoint_dir: Path) -> dict:
    """What `save` stored next to the weights (datasets, stage, ...)."""
    checkpoint = torch.load(checkpoint_dir / HEAD_FILE, map_location="cpu")
    return {key: value for key, value in checkpoint.items() if key != "head"}
