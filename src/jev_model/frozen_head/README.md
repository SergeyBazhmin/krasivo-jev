# frozen_head

A frozen Qwen decoder reads the sample as a text prompt, and an MLP on its last-token hidden state scores the
options. Only the MLP is trained, so the backbone runs once per prompt and its output is cached.

## How it works

The prompt (`jev_model/prompt.py`) numbers the options and ends on the answer cue:

```
{state}

Question: {question}
Options:
1) first option
2) second option
Answer:
```

The hidden state at the last token goes through `OptionHead` (`head.py`): LayerNorm, an MLP, and a linear layer
with 255 outputs. Output `i` is the logit of option `i`; the slots past a sample's option count are masked. The
head sees options only by their number, so a sample with more than 255 options is skipped.

A prompt over `cache.max_length` tokens loses the middle of its state. A sample whose question and options do not
fit on their own is skipped.

## Stages

| stage   | default | what it does |
|---------|---------|--------------|
| `embed` | yes     | Runs the backbone over every prompt and saves the last-token hidden states under `cache_dir/<cache slug>/<dataset>/<partition>.pt`. Training samples get `cache.views` option orders (the first is the source order), so no slot learns a prior for an answer. Runs with the same `[cache]` section share this cache. |
| `ce`    | yes     | Trains the head with soft cross-entropy on the cached features. |
| `rl`    | no      | Starts from the `ce` head and fine-tunes it with a policy gradient over perturbed logits, rewarded by a proper scoring rule (`jev_model/losses.py`). |

Each training stage keeps the head with the best macro validation NLL, fits one temperature on validation, and
writes `RUN/<stage>/head.pt`, `history.json` and `validation.json`.

## Usage

For Apple Silicon, use `uv sync --extra model-mac` and `configs/mac/frozen_head.toml`.
The CLI selects MPS automatically when available; `--device cpu` forces CPU.
See the root README for local experiments and the offline smoke script.

```bash
uv sync --extra model
uv run jev-model config frozen_head > my.json            # defaults, to edit
uv run jev-model train frozen_head -c my.json -s ce.lr=3e-4
uv run jev-model train frozen_head --run-dir runs/frozen_head/X --stage rl
uv run jev-model eval runs/frozen_head/X [--stage ce] [--partition validation]
uv run jev-model predict runs/frozen_head/X --question "..." --option yes --option no
```

## Config

`jev-model config frozen_head` prints every key. `configs/mac/frozen_head.toml` provides small local settings.
The main keys:

- `datasets`: built datasets to use; empty means every one under `data_dir`.
- `batch_tokens`: padded tokens per backbone pass in `embed`. Lower it when `embed` runs out of memory.
- `[cache]`: `model`, `layer`, `max_length`, `views`, `max_samples` (0 for all rows). This section is the cache key.
  The sample cap is per dataset and partition, applied after splitting and before expanding option views.
- `[ce]`, `[rl]`: steps, batch size, learning rate, dataset mixing (`alpha`), and the head shape
  (`width`, `depth`, `dropout`; used by `ce`, kept by `rl`).

## Files

- `config.py`: `CacheConfig`, `TrainConfig`, `FrozenHeadConfig`, `MAX_OPTIONS`.
- `backbone.py`: loads the frozen decoder; `embed` returns last-token hidden states.
- `cache.py`: builds and loads the feature cache.
- `head.py`: `OptionHead`.
- `train.py`: the `ce` and `rl` stages.
- `evaluate.py`: scores and temperature fitting on cached features.
- `predict.py`: `JevPredictor`, for `predict` and the UI.
- `pipeline.py`: `FrozenHeadModel`, the entry in the model registry.
