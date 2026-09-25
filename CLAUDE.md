# rujev

Training data for a "system one" model: a single encoder-only transformer that answers
multiple-choice questions in one forward pass. Every example has the shape

```
state -> question -> options -> label
```

The answer must follow from the `state` and `question` alone. The model is not meant to
store facts, so a sample that needs world, cultural or expert knowledge beyond the state
is noise here. It does not belong in the corpus, even if the source dataset is popular.

`src/jev_datasets/` turns Hugging Face datasets into this one format. There is no
training code in the repo yet.

## Sample contract

Defined in `src/jev_datasets/utils.py` (`SAMPLE_FEATURES`, `make_sample`):

| field      | type                          | meaning |
|------------|-------------------------------|---------|
| `state`    | `str`                         | the context the question is about (`""` if there is none) |
| `question` | `str`                         | the question itself |
| `options`  | `[{"id": str, "text": str}]`  | answer candidates. `text` is what the model reads and `id` is a stable key |
| `label`    | `[float32]`                   | probability distribution over `options`, the same length as `options` |

`make_sample(state, question, options, label)` takes either the `id` of the single correct
option (it becomes a one-hot) or a soft distribution, for example annotator vote shares in
`chaos_nli` or teacher targets in `jev_distill`.

## Layout

- `base.py`: `JevDataset(name, hf_path, hf_name)`. It lazily loads a `DatasetDict` into
  `.data`. Subclasses implement `prepare()`, which replaces `.data` with contract samples.
  Override `load()` when the source needs special loading (for example `bbh` loads one
  config per task and uses tasks as splits).
- `utils.py`: shared helpers. Use these instead of writing new ones:
  - `make_options(ids, texts=None)`, `make_option`
  - `pick_question(versions, seed)`: picks one of several question phrasings per sample.
    The choice is deterministic because it is seeded by the sample text.
  - `stable_shuffle(items, seed)`: deterministic shuffle for sources that always put the
    answer first
  - `explode(fn)`: adapts a `row -> [samples]` function to a batched `map`
  - `slugify`: builds an option id from text
  - `stratified_limit`: subsamples a split while keeping the label (argmax) distribution
- `constants.py`: `MAX_SAMPLES = 10_000` per split. `ROOT_DIR` is the default output
  directory. It resolves to `src/`, so built datasets land in `src/<name>/`.
- `cli.py`: the `jev` Typer CLI.
- `__init__.py`: the `datasets` registry, `{name: instance}`. Each new dataset must be
  imported and added here, in alphabetical order.
- Topic subpackages: `bugs`, `business`, `classification`, `distill`, `guardrails`,
  `knowledge`, `logic`, `nlu`, `reasoning`, `sentiment`, `spatial`, `toxicity`, `truthfulness`.
  `logic` holds synthetic deduction sets (ProofWriter, LogicNLI, FOLIO, ...) and `spatial`
  holds spatial-relation stories (StepGame, SpartQA, SpaRP).
- Many logic and spatial sources share data (for example, SpaRP includes StepGame, and
  RuleCollection bundles ProntoQA, LogicNLI, ProofWriter and more). Take each underlying
  set from only one source, so test rows don't leak into another dataset's train split.

## Commands

```bash
uv sync
uv run jev list                              # registered dataset names
uv run jev prepare sst2 anli                 # build specific datasets
uv run jev prepare all --max-samples 5000 --output-dir /path/to/out
```

`prepare` loads from the Hub, runs `prepare()`, does a stratified cap on each split, and
calls `save_to_disk(output_dir / name)`.

## Adding a dataset

Follow an existing converter. `sentiment/sst2.py` is the simplest; `reasoning/bbh.py`
and `knowledge/sciq.py` show parsing and shuffling.

1. **Check that the data is state-answerable before writing code.** Drop sources, subsets,
   tasks or rows that need outside knowledge, and say why in a comment. Examples: `sciq`
   keeps only rows with a support passage; `bbh` leaves out knowledge-heavy tasks;
   `jev_distill` keeps only `openjev_v2`. MMLU, ARC, OpenBookQA, CommonsenseQA,
   HellaSwag, WinoGrande and the preference/judge sets were removed for this reason.
2. Subclass `JevDataset`, then create a module-level instance `<name>_dataset = ...(name=..., hf_path=...)`.
3. In `prepare()`, `self.data.map(..., remove_columns=self.source_columns, features=SAMPLE_FEATURES)`.
   Always pass `features=SAMPLE_FEATURES`. Without it, `map` can inherit a source
   `ClassLabel` type for `label`.
4. Drop unlabelled splits, such as a `test` split with `label == -1`.
5. Give each dataset a `QUESTIONS` list of about 5 paraphrases and choose one per sample
   with `pick_question`, so the model does not overfit to a single fixed prompt.
6. If the source orders answers so that position gives away the label, shuffle the options
   deterministically (`stable_shuffle`, or `random.Random(seed)`). Use positional ids
   (`A`, `B`, ...) when option texts can repeat.
7. Option `text` should be human-readable (for example `"card arrival"` rather than
   `"card_arrival"`). Keep the raw value as the `id`.
8. Register the dataset in `src/jev_datasets/__init__.py`.

## Conventions

- Python 3.12+, managed with `uv`. Dependencies are `datasets` and `typer`.
- Everything must be deterministic across runs: seed from sample content, never from
  global randomness.
- Comments explain *why* a source is filtered or reshaped (quirks of the source, leakage,
  knowledge dependence). Keep them short, as in the existing modules.
- Lines are up to about 120 characters. Use type hints and `X | None` unions.
- There are no tests. To verify a converter, run `uv run jev prepare <name> --max-samples 50
  --output-dir <scratch dir>` and inspect a few rows with `datasets.load_from_disk`.
