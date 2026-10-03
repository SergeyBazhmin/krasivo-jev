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
- The none option replaces the true option's text in place (`none_prob`); kev's distractor augmentation is
  left out.
- Left out: the permutation KL, the prefix cache and serving paths, full-weight training, anchors, label
  smoothing, Brier and focal terms.

### None of the above

With probability `none_prob` (0.1) a training draw replaces the text of the true option with
`None of the above`, and that option keeps the label. Otherwise the model only ever sees closed sets and learns
that one of the listed options must be correct. A draw is left alone when the label is soft (no single true
option), when the sample already has that option, or when the longer sequence would not fit. Validation and
test rows are never changed, so the scores do not measure this behaviour.

## Stages

| stage | default | what it does |
|-------|---------|--------------|
| `ce`  | yes     | Trains the LoRA and the head with soft cross-entropy. Each draw shows the options in a fresh order, seeded by the sample's content and the draw number. |
| `rl`  | no      | Starts from the `ce` weights and fine-tunes both with a policy gradient over perturbed logits, rewarded by a proper scoring rule (`jev_model/losses.py`). |

`sampling` picks how training samples are drawn. `weighted` (the default) draws with replacement, each dataset
in proportion to `rows ** alpha`, and groups the draws by length within blocks of 2048 to pad less. `sized`
trains by passes over the data: each pass shuffles the whole training set and, in that order, fills each batch with
samples while its padded size stays within `batch_tokens`, so every sample is shown once per pass. Datasets are then
mixed through the loss instead: each sample is weighted so that its dataset's total weight is in proportion to
`rows ** alpha`, with a mean weight of 1. With `sized`, set `epochs` to train for that many passes; the number of
optimizer steps is then counted from the batches and replaces `steps`.

One optimizer step accumulates `accum` batches of at most `batch_tokens` padded tokens. Every `eval_every` steps the model is scored on `eval_samples` validation rows per dataset, and the
weights with the best macro NLL are kept. At the end the best weights are scored on the whole validation partition
and one temperature is fitted.
A stage writes `RUN/<stage>/adapter/` (the LoRA), `head.pt`, `history.json` and `validation.json`.

There is no feature cache: every training step and every evaluation runs the backbone. On CUDA the base weights
are loaded in bf16 and the forward pass uses bf16 autocast; the LoRA and the head stay in float32. The LoRA wraps
the whole causal LM, but the forward pass calls only the text model under it, so no vocabulary logits are
computed; older adapters (which wrap the text model) still load.

Most Qwen3.5 layers are linear attention (Gated DeltaNet). `transformers` runs them with the Triton kernel from
`flash-linear-attention`, which the `model` extra installs; without it they fall back to a reference PyTorch
implementation that is more than ten times slower, and a warning says so.

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
- `[ce]`, `[rl]`: `steps`, `batch_tokens`, `accum`, `lr`, `warmup`, `eval_every`, `eval_samples`, `sampling`, `epochs`, dataset
  mixing (`alpha`), `none_prob`. When training runs out of memory, halve `batch_tokens` and double `accum`.

## Files

- `config.py`: `TrainConfig`, `PointerConfig`.
- `encode.py`: `Encoder` tokenizes a sample and lays it out in a given option order.
- `model.py`: `PointerHead`, `PointerModel`, and `create` / `load` for the LoRA and the head.
- `data.py`: loads and tokenizes the partitions; `Pool` and `SizedEpochPool` draw training batches.
- `train.py`: the `ce` and `rl` stages.
- `evaluate.py`: scores and temperature fitting.
- `predict.py`: `PointerPredictor`, for `predict` and the UI.
- `pipeline.py`: `PointerJevModel`, the entry in the model registry.
