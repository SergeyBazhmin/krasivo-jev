# pointer

A Qwen decoder with a LoRA reads the sample marked up with delimiter tokens, and a pointer head scores each
option from the hidden state at its closing token. Adapted from [kev](https://github.com/jaredpalmer/kev)
(`kev/model.py`).

## How it works

The sequence (`encode.py`):

```
<state> state <q> question <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
```

The five delimiters reuse rarely used Qwen special tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`,
`<|box_end|>`, `<|fim_suffix|>`), so no embedding rows are added; the LoRA gives them their meaning. A `<|name|>`
in a sample's own text is rewritten before tokenizing, so it cannot forge an option boundary.

`PointerHead` (`model.py`) projects the hidden state at `<decide>` and the one at each `</opt>` to `head_dim` and
takes their scaled dot product as the option's logit. The head has no fixed number of slots, so there is no
option cap, and the option's own text drives its logit.

A sequence over `max_length` tokens loses the middle of its state. A sample whose question and options do not
fit on their own is skipped.

### Differences from kev

- One question per sequence. kev packs several questions after one state and needs a block-causal mask; here
  attention is plain causal, which also suits the linear-attention layers of Qwen3.5.
- No option isolation: an option attends to the options before it. The probabilities therefore depend on the
  option order. Training shuffles the options on every draw to keep the model from learning a position prior,
  but it does not make the output order-invariant.
- Left out: the permutation KL, the prefix cache and serving paths, full-weight training, anchors, none-option
  and distractor augmentation, label smoothing, Brier and focal terms.

## Stages

| stage | default | what it does |
|-------|---------|--------------|
| `ce`  | yes     | Trains the LoRA and the head with soft cross-entropy. Each draw shows the options in a fresh order, seeded by the sample's content and the draw number. |
| `rl`  | no      | Starts from the `ce` weights and fine-tunes both with a policy gradient over perturbed logits, rewarded by a proper scoring rule (`jev_model/losses.py`). |

One optimizer step accumulates `accum` batches of at most `batch_tokens` padded tokens. Every `eval_every` steps
the model is scored on `eval_samples` validation rows per dataset, and the weights with the best macro NLL are
kept. At the end the best weights are scored on the whole validation partition and one temperature is fitted.
A stage writes `RUN/<stage>/adapter/` (the LoRA), `head.pt`, `history.json` and `validation.json`.

There is no feature cache: every training step and every evaluation runs the backbone. The base weights are
loaded in float32 and the forward pass uses bf16 autocast on CUDA.

## Usage

```bash
uv sync --extra model
uv run jev-model config pointer > my.json                # defaults, to edit
uv run jev-model train pointer -c configs/pointer.toml -s ce.lr=1e-4
uv run jev-model train pointer --run-dir runs/pointer/X --stage rl
uv run jev-model eval runs/pointer/X [--stage ce] [--partition validation]
uv run jev-model predict runs/pointer/X --question "..." --option yes --option no
```

## Config

`configs/pointer.toml` lists every key with a comment. The main ones:

- `datasets`: built datasets to use; empty means every one under `data_dir`.
- `model`: the base decoder. Its tokenizer must have the delimiter tokens above.
- `lora`, `lora_targets`: the LoRA rank and where it goes (`all`: attention and MLP projections, `attn`, `qv`).
  On Qwen3.5, `all` and `attn` also cover the linear-attention projections.
- `special_embeddings`: also train the embedding rows of the five delimiters.
- `head_dim`, `max_length`, `eval_batch_tokens`.
- `[ce]`, `[rl]`: `steps`, `batch_tokens`, `accum`, `lr`, `warmup`, `eval_every`, `eval_samples`, dataset mixing
  (`alpha`). Lower `batch_tokens` when training runs out of memory.

## Files

- `config.py`: `TrainConfig`, `PointerConfig`.
- `encode.py`: `Encoder` tokenizes a sample and lays it out in a given option order.
- `model.py`: `PointerHead`, `PointerModel`, and `create` / `load` for the LoRA and the head.
- `data.py`: loads and tokenizes the partitions; `Pool` draws length-grouped training batches.
- `train.py`: the `ce` and `rl` stages.
- `evaluate.py`: scores and temperature fitting.
- `predict.py`: `PointerPredictor`, for `predict` and the UI.
- `pipeline.py`: `PointerJevModel`, the entry in the model registry.
