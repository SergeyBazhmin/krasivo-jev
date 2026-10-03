from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_backbone(model: str, device: str | torch.device) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """The frozen text decoder without its LM head. Qwen3.5 checkpoints are multimodal;
    `AutoModelForCausalLM` loads only the language model from them."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
    decoder = lm.model.to(device).eval().requires_grad_(False)
    return decoder, tokenizer


def length_batches(lengths: list[int], batch_tokens: int) -> Iterator[list[int]]:
    """Indices grouped by similar length, each batch at most `batch_tokens` once padded."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch: list[int] = []
    for i in order:
        if batch and lengths[i] * (len(batch) + 1) > batch_tokens:
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


@torch.inference_mode()
def embed(decoder: PreTrainedModel, pad_id: int, prompts: list[list[int]], layer: int = -1) -> torch.Tensor:
    """Hidden state at the last token of each prompt, [len(prompts), hidden], in float32.
    Padding goes on the right: every layer (attention or the linear-attention recurrence) is
    causal, so the pad tokens after a prompt cannot change its last real position."""
    device = next(decoder.parameters()).device
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        ids[row, : len(prompt)] = torch.tensor(prompt)
        mask[row, : len(prompt)] = 1
    ids, mask = ids.to(device), mask.to(device)
    out = decoder(input_ids=ids, attention_mask=mask, output_hidden_states=layer != -1, use_cache=False)
    hidden = out.last_hidden_state if layer == -1 else out.hidden_states[layer]
    last = mask.sum(dim=1) - 1
    return hidden[torch.arange(len(prompts), device=device), last].float().cpu()
