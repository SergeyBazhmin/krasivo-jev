# Finetuning on your own data

Training a `pointer` model means training a LoRA and a pointer head on top of a frozen Qwen3.5 base, on a mix of
built datasets. To teach it your task, build your data in the same format, put it next to the public datasets and
train a run on the mix.

> There is no warm start from an existing run: the `ce` stage always starts from the base model with a fresh LoRA
> and head. Train on your data **together with** the public datasets rather than on your data alone; a LoRA trained
> on one small dataset overfits it and loses the general skills (and the calibration) the mix gives.

## 1. Put your data in the sample format

Every row is one question:

| field | type | meaning |
|-------|------|---------|
| `state` | `str` | the text the question is about (`""` if there is none) |
| `question` | `str` | the question |
| `options` | `[{"id": str, "text": str}]` | two or more candidates; `text` is what the model reads |
| `label` | `[float]` | a probability per option, summing to 1: one-hot, or soft (e.g. annotator votes) |
| `type` | `str` | `noul` when the options are only yes/no, else `choice` (used by the comparison notebook) |

A dataset is a `DatasetDict` with exactly the splits `train`, `validation` and `test`, saved under `data/<name>/`.

### Quick route: save it yourself

```python
from datasets import Dataset, DatasetDict

from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

# your data: {"train": [(ticket text, team), ...], "validation": [...], "test": [...]}
my_splits = ...

def row(ticket: str, team: str) -> dict:
    options = make_options(["billing", "technical", "sales"], ["billing", "technical problem", "sales"])
    return make_sample(ticket, "Which team should handle this ticket?", options, team) | {"type": "choice"}

features = SAMPLE_FEATURES.copy()
features["type"] = SAMPLE_FEATURES["question"]   # a string column
splits = {name: [row(t, team) for t, team in rows] for name, rows in my_splits.items()}  # train/validation/test
DatasetDict({name: Dataset.from_list(rows, features=features) for name, rows in splits.items()}).save_to_disk(
    "data/my_tickets"
)
```

`make_sample` takes either the `id` of the correct option (it becomes a one-hot) or a list of probabilities.
Split by **source document**, not by row: if one ticket yields several questions, keep them all in one split, or
the test scores will be inflated.

### Proper route: a converter

If the data is on the Hugging Face Hub or will be rebuilt, add a converter to `src/jev_datasets/` and register it
(see [Adding a dataset](adding-a-dataset.md)). `jev prepare` then makes the splits for you: it keeps each context in one
split, drops exact duplicates, carves missing validation/test splits by hash, and caps each split.

```bash
uv run jev prepare my_tickets --max-samples 20000
```

### What makes good training data

- **The answer must follow from the state and the question.** Rows that need outside knowledge teach the model to
  guess. Drop them.
- **Paraphrase the question.** Use several phrasings and pick one per row deterministically (`pick_question`), so the
  model learns the task and not one prompt.
- **Readable option text.** `card arrival`, not `card_arrival`. Keep the raw value as the `id`.
- **Shuffle options when the source orders them** (answer always first): `stable_shuffle`, seeded by the row text.
- **Be deterministic.** Seed everything from the row content, never from global randomness, so a rebuild gives
  the same splits.

Check a build before training:

```python
from datasets import load_from_disk

data = load_from_disk("data/my_tickets")
print(data)                 # train / validation / test with the five columns
print(data["train"][0])
```

## 2. Build the public mix

```bash
uv run jev list                                   # registered datasets
uv run jev prepare all -j 8 --max-samples 25000   # what the released runs were built with
```

Every dataset lands in `data/<name>/`. Downloads come from the Hub; set `HF_TOKEN` for gated sources.

## 3. Configure the run

Start from the example config, which comments every key:

```bash
cp configs/pointer.toml configs/my_tickets.toml
uv run jev-model config pointer -c configs/my_tickets.toml    # print the resolved config
```

The settings that matter most:

| key | default | notes |
|-----|---------|-------|
| `datasets` | `[]` (every built dataset) | list yours with the public ones, or leave empty to use everything under `data_dir` |
| `model` | `unsloth/Qwen3.5-2B-Base` | the base decoder; its tokenizer must have the five delimiter tokens. Only Qwen3.5 exports to vLLM |
| `max_length` | 1024 | tokens per sequence. Raise it if your states are long; longer sequences cost memory and time |
| `lora`, `lora_targets` | 16, `all` | LoRA rank and where it goes (`all`, `attn`, `qv`) |
| `ce.sampling` | `weighted` | `weighted`: draw with replacement, each dataset in proportion to `rows ** alpha`. `sized`: passes over all rows |
| `ce.alpha` | 0.5 | weighted only: 1 = by size, 0 = uniform. Lower it to give a small dataset of yours more weight |
| `ce.steps` / `ce.epochs` | 5000 / 0 | `epochs` needs `sampling = "sized"` and replaces `steps` |
| `ce.batch_tokens`, `ce.accum` | 8192, 2 | padded tokens per forward pass, and passes per optimizer step |
| `ce.lr`, `ce.warmup` | 2e-4, 200 | |
| `ce.eval_every`, `ce.eval_samples` | 2500, 1000 | validation during training; the best macro-NLL weights are kept |
| `ce.none_prob`, `ce.drop_prob` | 0.1, 0.2 | augmentations: a *None of the above* option, and dropping wrong options so the option count varies |

The released run used the defaults with `ce.sampling = "sized"` and `ce.epochs = 2` on every built dataset
(`runs/pointer/20261003-174824/config.json`).

To make your dataset count for more without repeating it by hand, lower `ce.alpha` under `weighted` sampling. Under
`sized` sampling every row is shown once per pass, so a small dataset stays a small share of training.

## 4. Train

```bash
uv sync --extra model

# one GPU
uv run jev-model train pointer -c configs/my_tickets.toml --run-dir runs/pointer/my_tickets

# several GPUs (DDP); configure accelerate once per machine first
uv run accelerate config
uv run accelerate launch --multi_gpu --num_processes 4 -m jev_model.cli train pointer \
  -c configs/my_tickets.toml --run-dir runs/pointer/my_tickets
```

`-s key=value` overrides any key on top of the file (`-s ce.lr=1e-4 -s datasets=my_tickets,sst2,anli`).

With `N` GPUs each optimizer step sees `N` times more batches, and with `epochs` the step count is divided by `N`.
`steps`, `lr` and `warmup` are not rescaled for you. When training runs out of memory, halve `ce.batch_tokens` and
double `ce.accum`: the effective batch stays the same.

The run directory gets:

| path | what it is |
|------|------------|
| `config.json` | the resolved config; an existing run keeps it, so `--stage` can extend the run |
| `run.json` | the model and its finished stages |
| `ce/adapter/`, `ce/head.pt` | the LoRA and the head (with the settings and the fitted temperature) |
| `ce/history.json` | loss and validation scores over training |
| `ce/validation.json` | final scores on the whole validation partition |

At the end of the stage the best weights are scored on the full validation partition and one softmax temperature
is fitted on it, so the probabilities are calibrated.

`runs/` is tracked in git except the weights (`*.safetensors`, `*.pt`), so commit configs and scores, and move the
weights another way (`jev-model push`, below).

### Smoke test without a GPU

Before a long run, check the pipeline end to end on a tiny random decoder:

```bash
uv run jev-model train pointer -s model=<tiny local Qwen3.5-shaped decoder> -s ce.steps=40 \
  -s datasets=my_tickets --device cpu --run-dir /tmp/smoke
```

## 5. Evaluate

```bash
uv run jev-model eval runs/pointer/my_tickets                     # test partition
uv run jev-model eval runs/pointer/my_tickets --partition validation
```

Compare with the released run on the shared datasets to check that adding yours did not hurt the rest, using
`notebooks/compare_runs.ipynb` (set `RUNS` in its first cell). Every model reads the same test rows, so the numbers
are comparable.

## 6. Publish

```bash
# the Hub: only adapter/ and head.pt; JevHolder.from_pretrained("user/my-tickets") loads it
uv run jev-model push runs/pointer/my_tickets --repo user/my-tickets --private

# vLLM: a merged folder, see vllm.md
uv run jev-model export runs/pointer/my_tickets --out exports/my_tickets
```

## The `rl` stage

`rl` starts from the `ce` weights and fine-tunes with a policy gradient rewarded by a proper scoring rule
(`jev_model/losses.py`). It is kept in the code but not used: the runs here are trained with `ce` only. To try it:

```bash
uv run jev-model train pointer --run-dir runs/pointer/my_tickets --stage rl
uv run jev-model eval runs/pointer/my_tickets --stage rl
```
