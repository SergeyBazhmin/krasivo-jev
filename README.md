# krasivo-jev

Training data and models for a "system one" model: one forward pass answers a multiple-choice question from the
given context alone. Every sample has the shape `state -> question -> options -> label`; see `AGENTS.md` for the
sample contract and the conventions.

- `src/jev_datasets/`: converts Hugging Face datasets into that format (`jev` CLI).
- `src/jev_model/`: trains and evaluates models on the built datasets (`jev-model` CLI).
- `notebooks/compare_runs.ipynb`: compares evaluated runs with plots.

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # datasets only
uv sync --extra model                     # + training and evaluation
uv sync --extra model --extra notebook    # + the comparison notebook
uv sync --extra model --extra ui          # + the Streamlit UI
```

Secrets go in a `.env` file at the repo root (read on start, never written to a run's config), for example
`TYPESAFE_API_KEY=...` for the `typesafe` model.

### A local LLM server

The `zero_shot` baseline talks to an OpenAI-compatible server. `src/scripts/` starts one with
vLLM in Docker, on port 8080, with the Hugging Face cache mounted and `HF_TOKEN` passed through:

```bash
bash src/scripts/serve_gemma.sh   # google/gemma-4-26B-A4B-it, data parallel over 4 GPUs
bash src/scripts/serve_qwen.sh    # Qwen/Qwen3.5-2B, text only
```

`zero_shot` expects `http://localhost:8080/v1` by default.

## 1. Build the datasets

```bash
uv run jev list                                   # registered datasets
uv run jev prepare all                            # every dataset into data/<name>
uv run jev prepare sst2 anli --max-samples 5000   # some of them, a smaller cap per split
```

## 2. Train

```bash
uv run jev-model list                                   # models and their stages
uv run jev-model config pointer > my.json               # the defaults, to edit
uv run jev-model train pointer -c configs/pointer.toml -s ce.lr=1e-4 -s datasets=atis,banking77
uv run jev-model train pointer --run-dir runs/pointer/X --stage rl   # add a stage to an existing run
```

Config is layered: the dataclass defaults, then the `-c` file (JSON or TOML), then `-s key=value` overrides.
Each run lands in `runs/<model>/<timestamp>/` with its `config.json`, `run.json` and stage weights. `runs/` is
tracked in git so configs and scores are shared; weights (`*.safetensors`, `*.pt`) are ignored.

### Several GPUs

`pointer` trains with DDP under `accelerate`. Configure accelerate once per machine first (choose multi-GPU and
the number of GPUs), then launch:

```bash
uv run accelerate config    # once per machine; or `uv run accelerate config default` for the defaults
uv run accelerate launch --multi_gpu --num_processes 4 -m jev_model.cli train pointer -c configs/pointer.toml
```

### Baselines

`zero_shot` (a pretrained LM behind an OpenAI-compatible server such as vLLM) and `typesafe` (a hosted API)
train nothing: `train` only records the config, and `eval` scores them.

```bash
uv run jev-model train zero_shot --run-dir runs/zero_shot/chat   # server at http://localhost:8080/v1 by default
```

## 3. Evaluate

```bash
uv run jev-model eval runs/pointer/X                            # last stage, test partition
uv run jev-model eval runs/pointer/X --stage ce --partition validation
uv run jev-model predict runs/pointer/X --question "..." --option yes --option no
uv run jev-model ui                                             # try runs in the browser
```

`eval` prints per-dataset accuracy, F1, NLL and ECE and saves them to `RUN/<stage>-<partition>.json`
(`RUN/last-<partition>.json` without `--stage`).

## 4. Compare runs

```bash
uv run jupyter lab notebooks/compare_runs.ipynb
```

The notebook reads only the scores `eval` saved, so evaluate each run first. Set `RUNS`, `STAGES`, `PARTITION`,
`METRIC` and the head-to-head pair in its first cell. It shows a leaderboard, a per-dataset heatmap, two runs
head to head, scores by topic and by dataset type, and the datasets that split the runs or that none solves.

## Development

```bash
uv run ruff check --fix . && uv run ruff format .
```

There are no tests. To check a converter, run `uv run jev prepare <name> --max-samples 50 --output-dir <scratch>`
and inspect a few rows. To smoke-test a model without a GPU, train `pointer` with `-s model=<tiny local decoder>
-s ce.steps=40` on one or two small datasets. `AGENTS.md` describes how to add a dataset or a model.
