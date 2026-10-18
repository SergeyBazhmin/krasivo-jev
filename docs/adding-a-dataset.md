# Adding a dataset

A dataset enters training through a **converter**: a small class in `src/jev_datasets/` that loads a source (a Hugging
Face dataset or local files) and turns each row into one or more samples in the shared format. `jev prepare` then
splits, deduplicates, caps and saves it to `data/<name>/`, where training picks it up.

```
source rows ──prepare()──▶ samples ──regroup()──▶ train/validation/test ──add_type(), limit()──▶ data/<name>/
             (your code)              (automatic: disjoint contexts,          (automatic)
                                       duplicates dropped, missing splits carved)
```

For a one-off experiment you can skip the converter and save a `DatasetDict` yourself (see
[Finetuning](finetuning.md#quick-route-save-it-yourself)). Write a converter when the data should be rebuilt, shared
or kept in the mix.

## 0. Check the data first

The model reads; it does not remember. **Every answer must follow from the state and the question alone.** Before
writing code, look at 20 rows and ask: could a careful reader with no domain knowledge pick the label from what the
row shows?

- Keep: classification of the given text, reading comprehension, entailment, extraction, puzzles stated in full.
- Drop: trivia, exam questions without a passage, commonsense or cultural knowledge, "which answer do you prefer".
  MMLU, ARC, OpenBookQA, CommonsenseQA, HellaSwag, WinoGrande and the preference/judge sets were removed for this
  reason.
- Filter when only part qualifies, and say why in a comment: `sciq` keeps only rows with a support passage.

Also check that the source's test rows are not already in another dataset here. Many logic and spatial sources
bundle each other (SpaRP includes StepGame; RuleCollection bundles ProntoQA, LogicNLI, ProofWriter, ...). Take each
underlying set from one source only.

## 1. Write the converter

Pick the topic folder (`classification`, `nlu`, `sentiment`, `logic`, ...) and add a module. The simplest complete
example is `src/jev_datasets/sentiment/sst2.py`:

```python
from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is the sentiment of this movie review snippet?",
    "Is this excerpt from a movie review positive or negative?",
    "What opinion of the film does this phrase convey?",
    "What is the sentiment of this sentence from a film review?",
    "Does this snippet speak well or badly of the movie?",
]
ABOUT = [
    "You are given a single sentence from a movie review, lowercased and split into tokens. Decide whether the "
    "opinion of the film it conveys is positive or negative.",
    # ... about 5 paraphrases with the same definitions
]


class SST2Dataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()             # the source's ClassLabel names
        options = make_options(names)
        # the `test` split is unlabelled (label == -1)
        self.data.pop("test", None)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["sentence"], "Sentence"),   # the state
                pick_question(QUESTIONS, x["sentence"]),          # the question
                options,
                names[x["label"]],                                # the id of the correct option
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


sst2_dataset = SST2Dataset(name="sst2", hf_path="stanfordnlp/sst2")
```

### The pieces

| piece | what it does |
|-------|--------------|
| `JevDataset(name, hf_path, hf_name=None)` | `name` is the folder under `data/` and the CLI name; `hf_path`/`hf_name` go to `load_dataset`. `self.data` loads lazily |
| `type` | `DatasetType.NOUL` when the options are exactly English `yes`/`no`, else `DatasetType.CHOICE` (Russian «да»/«нет» is `choice`) |
| `prepare()` | replaces `self.data` with samples; drop unlabelled splits here |
| `make_sample(state, question, options, label)` | one row; `label` is the correct option's `id`, or a list of probabilities |
| `make_options(ids, texts=None)` | `[{"id", "text"}]`; `text` defaults to the id |
| `with_context(ABOUT, text, name, answers=None)` | the state: one `ABOUT` paraphrase (picked by `text`), then `name:` and the text. `answers` adds a list explaining what each answer means |
| `pick_question(QUESTIONS, seed)` | one paraphrase per row, deterministic |
| `stable_shuffle(items, seed)` | a deterministic shuffle |
| `text_choice(state, question, answer, distractors, seed)` | a sample from an answer and distractor texts: deduplicated, shuffled, positional ids |
| `explode(fn)` | adapts `row -> [samples]` to a batched `map`, for several samples per row |
| `slugify(text)` | an option id from text |

### Rules that keep the data useful

1. **Always pass `features=SAMPLE_FEATURES` to `map`.** Without it, `label` can inherit the source's `ClassLabel`
   type and break.
2. **Put the task's conventions in the state.** The model stores no facts, so whatever a human annotator was told
   (label definitions, what "neutral" means, that the text is lowercased) goes into `ABOUT`, or into `answers` for
   per-label meanings. Write about 5 paraphrases of `ABOUT` that keep the same definitions.
3. **About 5 question paraphrases**, picked with `pick_question`, so the model learns the task and not one prompt.
4. **Readable option text.** `"card arrival"`, not `"card_arrival"`; keep the raw value as the `id`:
   `make_options(names, [n.replace("_", " ") for n in names])`.
5. **Shuffle when position gives the answer away.** If the source always lists the answer first, shuffle with
   `stable_shuffle` or `random.Random(seed)`, seeded by the row's text. Use positional ids (`A`, `B`, ...) when
   option texts can repeat.
6. **Everything deterministic.** Seed from row content, never from global randomness or the row index, so a rebuild
   gives the same rows and splits.
7. **Explain filters in short comments**: why a split, task or row is dropped (unlabelled, needs knowledge, leaks).

### Common variations

**Several questions per row** (one per label, one per span, ...): return a list and wrap it in `explode`, with
`batched=True`. Simplified from `sentiment/go_emotions.py`, which asks a yes/no question per emotion:

```python
def noul_samples(row: dict) -> list[dict]:
    state = with_context(NOUL_ABOUT, row["text"], "Comment")
    return [
        make_sample(state, pick_question(NOUL_QUESTIONS, row["text"] + e).format(emotion=e), YES_NO, answer)
        for e, answer in questions_for(row)
    ]

self.data = self.data.map(
    explode(noul_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
)
```

Balance yes and no: asking about every label of a 27-way task makes most answers "no". `go_emotions_noul` asks about
the true emotions plus two sampled wrong ones.

**One document, several states** (passages of an article, turns of a dialogue): emit a `group` column keying the
document, so `regroup()` keeps the whole document in one split. Use `GROUPED_FEATURES` instead of `SAMPLE_FEATURES`;
the column is removed after regrouping. See `retrieval/wikiqa.py`:

```python
samples.append({**sample, "group": row["query"]})
...
self.data = self.data.map(
    explode(make_samples, GROUPED_FEATURES), batched=True, remove_columns=self.source_columns,
    features=GROUPED_FEATURES,
)
```

Without a `group`, rows are grouped by their state (or question, when the state is empty), which already keeps
several questions about one passage together.

**Soft labels** (annotator votes): pass the distribution, e.g. `make_sample(state, q, options, [0.6, 0.3, 0.1])`, as
`nlu/chaos_nli.py` does. It must have one value per option and sum to 1.

**Special loading**: override `load()` and return a `DatasetDict`. For a Hub dataset whose main branch is a loading
script (`nlu/dream.py`):

```python
def load(self) -> DatasetDict:
    return load_dataset(self.hf_path, revision="refs/convert/parquet")
```

For local files:

```python
def load(self) -> DatasetDict:
    return load_dataset("json", data_files={"train": "raw/tickets/train.jsonl", "test": "raw/tickets/test.jsonl"})
```

Raw archives and official JSON releases (no Hub copy) are handled in `structured/source.py` (`grouped_records`,
`map_records`); see `structured/totto.py`.

### Splits are handled for you

Return whatever splits the source has. `jev prepare` then:

- renames them (`dev`/`val` become `validation`; unknown names join `train`);
- moves every context group whole into the first split it appears in, test first, then validation, then train, so
  no test context is trained on;
- drops exact duplicate rows, keeping the held-out copy;
- carves missing splits by a hash of the group: 5% of train each for a missing validation or test, or 80/10/10 when
  there is no train split;
- stamps `type` on every row and caps each split with a seeded random sample.

So do not re-split or deduplicate in the converter, and do not add `type` yourself.

## 2. Register it

Import the instance in `src/jev_datasets/__init__.py` and add it to the `datasets` tuple, both in alphabetical
order:

```python
from jev_datasets.sentiment.my_reviews import my_reviews_dataset
...
datasets = {
    dataset.name: dataset
    for dataset in (
        ...
        my_reviews_dataset,
        ...
    )
}
```

Then add a row to the table in `src/jev_datasets/README.md` (topic, name, type, source and any filter worth knowing).

## 3. Build and inspect a small sample

```bash
uv run jev list | grep my_reviews
uv run jev prepare my_reviews --max-samples 50 --output-dir /tmp/jev-check
```

`prepare` logs the rows it moved between splits, duplicates dropped and splits carved; a large "moved" count means
the source's splits share contexts. Then read some rows:

```python
from datasets import load_from_disk

data = load_from_disk("/tmp/jev-check/my_reviews")
print(data)                                  # train / validation / test
for row in data["train"].select(range(3)):
    print(row["state"], "\n", row["question"], row["options"], row["label"], row["type"], "\n---")
```

Check that:

- the state holds everything the answer needs, and the instruction reads well;
- option texts are human-readable and the label points at the right one;
- the answer's position varies across rows (no "always A");
- labels are not wildly unbalanced, or are balanced on purpose;
- a `noul` dataset really has only `yes`/`no` options.

Then lint:

```bash
uv run ruff check --fix . && uv run ruff format .
```

## 4. Build it for real and train

```bash
uv run jev prepare my_reviews --max-samples 20000      # into data/my_reviews
```

`pointer` trains on every dataset under `data/` when `datasets` is empty, so the new one joins the next run. To train
on a chosen mix, list it: `-s datasets=my_reviews,sst2,anli`. See [Finetuning](finetuning.md) for the rest.
