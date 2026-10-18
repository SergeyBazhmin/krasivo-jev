# krasivo-jev

Training data for a "system one" model: a single encoder-only transformer that answers
multiple-choice questions in one forward pass. Every example has the shape

```
state -> question -> options -> label
```

The answer must follow from the `state` and `question` alone. The model is not meant to
store facts, so a sample that needs world, cultural or expert knowledge beyond the state
is noise here. It does not belong in the corpus, even if the source dataset is popular.

`src/jev_datasets/` turns Hugging Face datasets into this one format. `src/jev_model/` trains
and evaluates models on the built datasets.

## Sample contract

Defined in `src/jev_datasets/utils.py` (`SAMPLE_FEATURES`, `make_sample`):

| field      | type                          | meaning |
|------------|-------------------------------|---------|
| `state`    | `str`                         | the context the question is about (`""` if there is none) |
| `question` | `str`                         | the question itself |
| `options`  | `[{"id": str, "text": str}]`  | answer candidates. `text` is what the model reads and `id` is a stable key |
| `label`    | `[float32]`                   | probability distribution over `options`, the same length as `options` |
| `type`     | `str`                         | the dataset's `DatasetType`: `noul` (options are only yes/no) or `choice` |

`type` is not produced by `prepare()`. `JevDataset.add_type()` stamps it on every row from the
class attribute `type`, which each converter must set.

When one source document yields several different states (passages, sentences, dialogue turns), `prepare()` also
emits a `group` column (`GROUPED_FEATURES`) keying the document, so `regroup()` keeps it in one split. `regroup()`
removes the column. `structured/source.py` (`grouped_records`, `map_records`) does this for raw-file sources.

`make_sample(state, question, options, label)` takes either the `id` of the single correct
option (it becomes a one-hot) or a soft distribution, for example annotator vote shares in
`chaos_nli`.

## Layout

- `base.py`: `JevDataset(name, hf_path, hf_name)`. It lazily loads a `DatasetDict` into
  `.data`. Subclasses implement `prepare()`, which replaces `.data` with contract samples.
  Override `load()` when the source needs special loading (for example `bbh` loads one
  config per task and uses tasks as splits).
- `utils.py`: shared helpers. Use these instead of writing new ones:
  - `make_options(ids, texts=None)`, `make_option`
  - `pick_question(versions, seed)`: picks one of several question phrasings per sample.
    The choice is deterministic because it is seeded by the sample text.
  - `with_context(about, text, name, answers)`: puts `text` under the dataset's instruction. `about` is a list of
    paraphrases; one is picked by `text`, so every row on the same context gets the same state.
    `pick_about(versions, seed)` does the same pick for states built by hand.
  - `stable_shuffle(items, seed)`: deterministic shuffle for sources that always put the
    answer first
  - `explode(fn)`: adapts a `row -> [samples]` function to a batched `map`
  - `slugify`: builds an option id from text
  - `random_limit`: caps a split with a seeded uniform random sample
- `constants.py`: `MAX_SAMPLES = 10_000` per split. `ROOT_DIR` is the default output
  directory. It resolves to `data/` at the repo root, so built datasets land in `data/<name>/`.
- `cli.py`: the `jev` Typer CLI.
- `__init__.py`: the `datasets` registry, `{name: instance}`. Each new dataset must be
  imported and added here, in alphabetical order.
- Topic subpackages: `bugs`, `business`, `classification`, `guardrails`, `knowledge`,
  `logic`, `nlu`, `reasoning`, `sentiment`, `spatial`, `toxicity`, `truthfulness`.
  `logic` holds synthetic deduction sets (ProofWriter, LogicNLI, FOLIO, ...) and `spatial`
  holds spatial-relation stories (StepGame, SpartQA, SpaRP).
- Benchmark datasets (`benchmark = True` on the class) are held out. They are saved with the usual
  train/validation/test splits. Training skips them unless `pointer`'s `train_on_benchmark` is set (a final model on
  all data, which then trains on their train split as usual). At `eval`, every model is scored on every benchmark,
  on the asked-for partition (test by default) like any other dataset.
  The benchmarks:
  - OpenDecisions (`business/open_decisions.py`), split by type into `open_decisions_noul` and
    `open_decisions_choice` (its `score` items are dropped). It keeps each item's own instructions and option
    order, with no paraphrases or shuffling, so every model sees the same item.
  - one converter file per other topic, drawn by `random.Random(f"eval_only:{topic}")`: `github_issue_type`,
    `hwu64`, `jailbreak_classification`, `ropes`, `ar_lsat`, `rublimp`, `logiqa`, `nomiracl*`, `tweet_sentiment`,
    `stepgame`, `webnlg_en`, `russian_toxicity`, `vitaminc`.
- Many logic and spatial sources share data (for example, SpaRP includes StepGame, and
  RuleCollection bundles ProntoQA, LogicNLI, ProofWriter and more). Take each underlying
  set from only one source, so test rows don't leak into another dataset's train split.

## Commands

```bash
uv sync
uv run jev list                              # registered dataset names
uv run jev prepare sst2 anli                 # build specific datasets
uv run jev prepare all --max-samples 5000 --output-dir /path/to/out
uv run jev prepare all -j 8                  # 8 datasets at once, one process each
```

`prepare` loads from the Hub and runs `prepare()`. It then calls `regroup()`, which groups rows by their `group` column
if there is one, else by a hash of their context (`state or question`), and moves each group whole into the first
split it appears in, held-out splits first (test, validation, train). A context is never in two splits. Exact duplicate
rows are dropped, keeping the held-out copy; no other row is lost. Split names are first made canonical (`dev`/`val` ->
`validation`, unknown -> `train`), and afterwards missing splits are carved by a hash of the group (`split_missing`):
`HOLDOUT_PERCENT` (5%) of train each for a missing validation/test, or an 80/10/10 re-split of every row when the
source has no train split. So every saved dataset has exactly `train`, `validation` and `test`. Converters don't need to
handle this themselves. Finally it caps each split with a seeded random sample and calls
`save_to_disk(output_dir / name)`.

## Adding a dataset

Follow an existing converter. `sentiment/sst2.py` is the simplest; `reasoning/bbh.py`
and `knowledge/sciq.py` show parsing and shuffling.

1. **Check that the data is state-answerable before writing code.** Drop sources, subsets,
   tasks or rows that need outside knowledge, and say why in a comment. Examples: `sciq`
   keeps only rows with a support passage; `bbh` leaves out knowledge-heavy tasks. MMLU,
   ARC, OpenBookQA, CommonsenseQA, HellaSwag, WinoGrande and the preference/judge sets were removed for this reason.
2. Subclass `JevDataset`, then create a module-level instance `<name>_dataset = ...(name=..., hf_path=...)`.
3. In `prepare()`, `self.data.map(..., remove_columns=self.source_columns, features=SAMPLE_FEATURES)`.
   Always pass `features=SAMPLE_FEATURES`. Without it, `map` can inherit a source
   `ClassLabel` type for `label`.
4. Drop unlabelled splits, such as a `test` split with `label == -1`.
5. Give each dataset a `QUESTIONS` list of about 5 paraphrases and choose one per sample
   with `pick_question`, so the model does not overfit to a single fixed prompt. Likewise give its
   instruction (`ABOUT`) about 5 paraphrases that keep the same definitions and conventions.
6. If the source orders answers so that position gives away the label, shuffle the options
   deterministically (`stable_shuffle`, or `random.Random(seed)`). Use positional ids
   (`A`, `B`, ...) when option texts can repeat.
7. Option `text` should be human-readable (for example `"card arrival"` rather than
   `"card_arrival"`). Keep the raw value as the `id`.
8. Register the dataset in `src/jev_datasets/__init__.py`.

## Models (`src/jev_model/`)

Mirrors `jev_datasets`: a registry of models, each with its own pipeline, driven by one CLI.

- `base.py`: `JevModel[Config](name, description)`. A model sets `config_class` (a dataclass with
  defaults for every field), `stages` (its ordered pipeline) and `default_stages`, and implements
  `run_stage`, `evaluate` and `predictor`. `train()` runs stages in pipeline order and records
  finished ones in `run.json`; the resolved config goes to `config.json`.
- `config.py`: config = dataclass defaults, then a JSON/TOML file (`-c`), then `-s key=value`
  overrides with dotted keys. Values parse as JSON, else string; `list` fields also take `a,b`. It also loads
  `.env` (git-ignored) into the environment, so secrets such as `TYPESAFE_API_KEY` never land in `config.json`.
  `configs/pointer.toml` is the example config, with every key commented.
- `data.py`: `resolve_names` and `partition` (the saved train/validation/test splits; it fails on datasets built
  before `jev prepare` made all three). Every model reads data through it, so test sets are the same rows.
  Training goes through `train_names`, which leaves out benchmark datasets. `pointer` evaluates its training
  datasets plus the benchmarks it didn't train on; `typesafe` scores its datasets plus every benchmark.
- `metrics.py`: scores over padded option logits (`scores`, `with_macro`, `fit_temperature`).
- `prompt.py`: the numbered-option prompt for decoder backbones.
- `distributed.py`: helpers for training in several processes under `accelerate launch` (rank checks, barriers,
  one shared run directory). A model that can train with DDP sets `multi_gpu = True`; the others refuse to run
  in more than one process. Only the main process writes run files.
- `constants.py`: `DATA_DIR`, `CACHE_DIR/<model>/...` (artefacts runs can share), `RUNS_DIR/<model>/<run>`.
  Runs are tracked in git (configs, scores, logs) except their weights (`*.safetensors`, `*.pt`).
- `__init__.py`: the `models` registry, alphabetical.
- `ui/app.py`: the Streamlit page behind `jev-model ui`. It picks a run, fills the form by hand or from a
  built dataset row, and calls `JevModel.predictor`, so new models need no UI code.
- `pointer/`: Qwen decoder with a LoRA + pointer head, after `jaredpalmer/kev` but with one question per
  sequence and no option isolation, so attention stays plain causal. The input is
  `<state> .. <q> .. <opt> .. </opt> .. <decide>` (delimiters reuse Qwen special tokens, `encode.py`); an
  option's logit matches the hidden state at its `</opt>` against the one at `<decide>`, so there is no option
  cap. Stages `ce` (options shuffled on every draw) -> `rl` (opt-in).
  There is no feature cache: every step runs the backbone.
  Each stage writes `RUN/<stage>/adapter` and `head.pt`. It trains on several GPUs with DDP under `accelerate launch`.
- `zero_shot/`: a baseline that uses a pretrained LM with no training. An OpenAI-compatible server (a local vLLM)
  replies with the number of an option. `train` only records the config, and `eval` scores it.
- `src/scripts/`: `serve_gemma.sh` (`google/gemma-4-26B-A4B-it`) and `serve_qwen.sh` (`Qwen/Qwen3.5-2B`) start
  an OpenAI-compatible vLLM server in Docker on `http://localhost:8080/v1`, for the `zero_shot` model.
  `uv run serve EXPORT_DIR` (`src/jev_vllm/serve.py`) builds `docker/vllm.Dockerfile` (vLLM 0.17.1 plus this package,
  so the `jev_vllm` plugins load) and serves a `jev-model export` folder on `http://localhost:8080/pooling`.
- `typesafe/`: TypeSafe's hosted `jev` model (`typesafe/jev-1.13`) through its `system_one` API (`typesafe-sdk`).
  Each sample is one `noul` (yes/no options) or `choice` question, and the returned probabilities are the prediction,
  so `nll` and `ece` are real. The key comes from `TYPESAFE_API_KEY`. Like `zero_shot`, `train` only records the config.
- `losses.py`: the lr schedule and the RL loss over option logits, used by `pointer`.
- `system_one.py`: `JevHolder.from_pretrained(run dir | stage dir | HF repo id)` and `.system_one(state, questions)`
  with named `Noul`/`Choice` questions, TypeSafe-style, answered offline in one batch. `jev-model push RUN --repo
  user/name` uploads a stage's `adapter/` and `head.pt`; the base model is fetched by name.

```bash
uv sync --extra model
uv run jev-model list                                   # models and their stages
uv run jev-model config pointer > my.json               # defaults, to edit
uv run jev-model train pointer -c my.json -s ce.lr=1e-4 -s datasets=atis,banking77
uv run jev-model train pointer --run-dir runs/pointer/X --stage rl   # extend a run
uv run accelerate launch --multi_gpu --num_processes 4 -m jev_model.cli train pointer -c configs/pointer.toml  # DDP
uv run jev-model eval runs/pointer/X [--stage ce] [--partition validation]
uv run jev-model predict runs/pointer/X --question "..." --option yes --option no
uv sync --extra model --extra ui && uv run jev-model ui   # try runs in the browser
uv sync --extra notebook && uv run jupyter lab notebooks/compare_runs.ipynb   # compare runs with plots
```

`notebooks/compare_runs.ipynb` compares runs from the scores `eval` saved (`RUN/<stage>-<partition>.json`): a
leaderboard, a per-dataset heatmap, two runs head to head, and scores by topic and by `type`. It only reads files,
so evaluate a run before comparing it.

### Adding a model

1. Create `src/jev_model/<name>/` with a `config.py` holding its config dataclass(es), and a
   `pipeline.py` with a `JevModel` subclass and a module-level `<name>_model = ...(name=..., description=...)`.
2. Import torch/transformers and the model's own modules inside `run_stage`/`evaluate`/`predictor`,
   not at the top of `pipeline.py`/`config.py`, so the CLI loads without every model's dependencies.
3. Read data through `jev_model.data` and report with `jev_model.metrics`, so numbers are comparable
   across models. Keep outputs inside the run directory; put only cross-run caches under `CACHE_DIR/<name>`.
4. Register it in `src/jev_model/__init__.py`. Add new dependencies to the `model` extra.
5. Add a `README.md` to the model's directory: how it works, its stages, usage, config and files.

## Conventions

- Python 3.12+, managed with `uv`. Dependencies are `datasets`, `loguru` and `typer`; `jev_model` also needs
  the `model` extra (torch, transformers, tqdm, peft, accelerate, flash-linear-attention, python-dotenv,
  typesafe-sdk). The `ui` extra adds streamlit; the `notebook` extra adds jupyterlab, pandas and matplotlib.
- Everything must be deterministic across runs: seed from sample content, never from
  global randomness.
- Comments explain *why* a source is filtered or reshaped (quirks of the source, leakage,
  knowledge dependence). Keep them short, as in the existing modules.
- Lines are up to 120 characters. Use type hints and `X | None` unions. Run `uv run ruff check --fix . && uv run ruff format .`
  before committing (config in `pyproject.toml`).
- There are no tests. To smoke-test a model pipeline without a GPU, save a tiny random decoder
  locally and train `pointer` with `-s model=<dir> -s ce.steps=40` on one or two
  small datasets.
- To verify a converter, run `uv run jev prepare <name> --max-samples 50
  --output-dir <scratch dir>` and inspect a few rows with `datasets.load_from_disk`.
