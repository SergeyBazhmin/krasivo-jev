# krasivo-jev

Training data and models for a "system one" model: one forward pass answers a multiple-choice question from the
given context alone. Every sample has the shape `state -> question -> options -> label`; see `AGENTS.md` for the
sample contract and the conventions.

- `src/jev_datasets/`: converts Hugging Face datasets into that format (`jev` CLI; [README](src/jev_datasets/README.md)).
- `src/jev_model/`: trains and evaluates models on the built datasets (`jev-model` CLI; [README](src/jev_model/README.md)).
- `docs/`: guides to using, serving (vLLM) and finetuning the model, and adding datasets ([index](docs/README.md)).
- `notebooks/compare_runs.ipynb`: compares evaluated runs with plots. `system_one.ipynb` and `system_one_vllm.ipynb`
  ask a model questions in process and through vLLM.
- `cards/`: the Hugging Face model cards of the released runs (`make_card.py` builds plots, tables and README).

## Models

Two `pointer` runs (Qwen3.5-2B-Base + LoRA + pointer head) are released. Each is a `jev-model export` folder with
merged bf16 weights for vLLM and the LoRA stage for `JevHolder`. Their cards pin this repo at the commit they were
released from.

| model | run | trained on | held-in test acc | benchmark test acc | card |
|---|---|---|---:|---:|---|
| `pointer-bf16` | [`20261018-160511`](runs/pointer/20261018-160511) | all 100 datasets, benchmarks included (the final model) | 0.877 | 0.791 (seen) | [card](cards/pointer-20261018-160511/README.md) |
| `pointer-no-bench-bf16` | [`20261017-192041`](runs/pointer/20261017-192041) | the 80 core datasets | 0.879 | 0.640 (unseen) | [card](cards/pointer-20261017-192041/README.md) |

Accuracies are macro means over datasets. The no-bench run's benchmark score is the estimate for new tasks.

```python
from jev_model.system_one import Choice, JevHolder, Noul

holder = JevHolder.from_pretrained("pointer-bf16")  # the export folder
answers = holder.system_one(
    state="I was charged twice for my order. Please help.",
    questions={"billing": Noul(instructions="Is this about billing?"),
               "tone": Choice(instructions="What is the tone?", criteria={"angry": None, "calm": "polite"})},
)
```

```bash
uv run serve pointer-bf16    # vLLM + jev_vllm plugins in Docker, http://localhost:8080/pooling
```

The served model is called with `curl` or with `jev_vllm.client.JevClient`, a drop-in for `typesafe-sdk`'s
`TypeSafeClient` ([Serving with vLLM](docs/vllm.md)). To publish a run: `uv run jev-model export runs/pointer/X --out OUT`,
then `uv run jev-model push OUT --repo user/name`.

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

## 1. Build the datasets

```bash
uv run jev list                                   # registered datasets
uv run jev prepare all                            # every dataset into data/<name>
uv run jev prepare sst2 anli --max-samples 5000   # some of them, a smaller cap per split
uv run jev prepare all -j 8                       # 8 datasets at once, one process each
```

The default cap is 10,000 rows per split (a seeded random sample). The `pointer` runs in `runs/` were built with
`--max-samples 25000`.

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

A full `pointer` run on every dataset needs at least 4 A100s and takes about 12–13 hours on 4. To repeat a released
run, pass its `runs/pointer/<run>/config.json` as `-c`, with `-s data_dir=data` (the saved path is the training
machine's).

### Baselines

`typesafe` (a hosted API) trains nothing: `train` only records the config, and `eval` scores it. The `zero_shot`
code (a pretrained LM behind an OpenAI-compatible server, started by `src/scripts/serve_gemma.sh` or `serve_qwen.sh`)
stays in `src/jev_model/zero_shot/` but is not registered.

## 3. Evaluate

```bash
uv run jev-model eval runs/pointer/X                            # last stage, test partition
uv run jev-model eval runs/pointer/X --stage ce --partition validation
uv run jev-model predict runs/pointer/X --question "..." --option yes --option no
uv run jev-model ui                                             # try runs in the browser
```

`eval` prints per-dataset accuracy, F1, NLL and ECE and saves them to `RUN/<stage>-<partition>.json`
(`RUN/last-<partition>.json` without `--stage`).

All scores in `runs/` are on the test partition of each dataset, so every model is compared on the same rows.

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
