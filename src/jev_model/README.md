# jev_model

Trains and evaluates models on the datasets `jev_datasets` builds. A registry of models, each with its own pipeline
of stages, driven by one CLI (`jev-model`). Every model reads data and reports scores through the same shared
modules, so their numbers are comparable on the same test rows.

## Models

| model       | stages      | what it is |
|-------------|-------------|------------|
| `pointer`   | `ce*` -> `rl` | Qwen decoder with a LoRA and a pointer head over the option tokens ([README](pointer/README.md)) |
| `typesafe`  | `api*`      | TypeSafe's hosted `jev` model through its `system_one` API ([README](typesafe/README.md)) |
| `zero_shot` | `generate*` | a pretrained chat LM on an OpenAI-compatible server, replying with an option number ([README](zero_shot/README.md)) |

`*` runs by default; later stages are opt-in. `typesafe` and `zero_shot` train nothing: `train` only records the
config, and `eval` scores them.

## Usage

```bash
uv sync --extra model
uv run jev-model list                                   # models and their stages
uv run jev-model config pointer > my.json               # the defaults, to edit
uv run jev-model train pointer -c configs/pointer.toml -s ce.lr=1e-4 -s datasets=atis,banking77
uv run jev-model train pointer --run-dir runs/pointer/X --stage rl   # add a stage to an existing run
uv run accelerate launch --multi_gpu --num_processes 4 -m jev_model.cli train pointer -c configs/pointer.toml
uv run jev-model eval runs/pointer/X [--stage ce] [--partition validation]
uv run jev-model predict runs/pointer/X --question "..." --option yes --option no
uv sync --extra model --extra ui && uv run jev-model ui   # try runs in the browser
```

## Runs

`train` writes `runs/<model>/<timestamp>/` (or `--run-dir`):

- `config.json`: the resolved config. An existing run keeps it, so `--stage` can extend the run.
- `run.json`: the model and its finished stages, in pipeline order.
- `<stage>/...`: what each stage saves (weights, logs).
- `<stage>-<partition>.json`: scores from `eval` (`last-<partition>.json` without `--stage`).

Runs are tracked in git except their weights (`*.safetensors`, `*.pt`). Artefacts that several runs can share go
under `cache/<model>/`.

## Config

A model's config is a dataclass with a default for every field. A run's config is those defaults, then a JSON or
TOML file (`-c`), then `-s key=value` overrides with dotted keys (`ce.lr=3e-4`). Values parse as JSON, else as a
string; `list` fields also take `a,b`. A `.env` at the repo root is loaded into the environment, so secrets such as
`TYPESAFE_API_KEY` never land in `config.json`. `configs/pointer.toml` is an example with every key commented.

## Data and metrics

- `data.py`: `resolve_names` (every built dataset in `data/` when none are named) and `partition`, which maps
  source splits to train/validation/test and carves missing ones from train by a content hash (5% each). Rows that
  share a state land in the same partition.
- `metrics.py`: `scores` over padded option logits: accuracy and macro precision/recall/F1 against the argmax of
  the label, soft NLL, and top-1 ECE. `with_macro` adds the unweighted mean over datasets; `fit_temperature`
  calibrates logits.

`eval` prints one line per dataset plus `macro`. `notebooks/compare_runs.ipynb` compares evaluated runs.

## Files

- `base.py`: `JevModel[Config]`, the base class: `config_class`, `stages`, `default_stages`, `multi_gpu`, and
  `run_stage`, `evaluate`, `predictor` to implement. `train()` runs stages in pipeline order.
- `cli.py`: the `jev-model` CLI.
- `config.py`: config loading and overrides.
- `constants.py`: `DATA_DIR`, `CACHE_DIR`, `RUNS_DIR`, `PARTITIONS`.
- `data.py`, `metrics.py`: shared data splits and scores.
- `distributed.py`: DDP helpers for `accelerate launch` (rank, barrier, one shared run directory); only the main
  process writes run files.
- `losses.py`: the lr schedule and the RL loss over option logits.
- `prompt.py`: the numbered-option prompt for decoder backbones, and middle truncation of long states.
- `ui/app.py`: the Streamlit page behind `jev-model ui`; it works through `JevModel.predictor`, so new models need
  no UI code.
- `pointer/`, `typesafe/`, `zero_shot/`: the models.

## Adding a model

See `AGENTS.md`. In short: create `<name>/config.py` and `<name>/pipeline.py` with a `JevModel` subclass, import
heavy dependencies inside its methods, read data through `jev_model.data` and score with `jev_model.metrics`,
register it in `__init__.py`, and add a `README.md` to its directory.
