# Using a trained model

A trained `pointer` model takes a **state** (the text the question is about), a **question** and two or more
**options**, and returns a probability for each option. It never generates text, so one question costs one forward
pass, however many options it has.

## From Python: `JevHolder`

`jev_model.system_one` asks named questions about one state, in the shape of TypeSafe's `system_one` API. There are
two kinds of question:

- `Noul(instructions)`: a yes/no question. The answer is `NoulAnswer.noul`, the probability of *yes*.
- `Choice(instructions, criteria)`: one label out of two or more. `criteria` maps each label to a description of
  when it applies, or to `None` when the label speaks for itself. The answer is a `ChoiceAnswer` with `choice` (the
  likeliest label), `confidence` (its probability) and `probabilities` (every label's).

```python
from jev_model.system_one import Choice, JevHolder, Noul

holder = JevHolder.from_pretrained("runs/pointer/20261003-174824")

answers = holder.system_one(
    state="Hi, I ordered a blender two weeks ago (order #4821) and it still hasn't arrived. "
    "I'd like a refund if it can't be delivered by Friday.",
    questions={
        "complaint": Noul("Is the customer complaining?"),
        "gave_order_id": Noul("Does the message include an order number?"),
        "topic": Choice(
            "What is the message mainly about?",
            {
                "delivery": "shipping, tracking or a late or lost parcel",
                "billing": "charges, invoices or payment methods",
                "account": None,
            },
        ),
    },
)
# the numbers below are illustrative
answers["complaint"].noul        # 0.97
answers["topic"].choice          # "delivery"
answers["topic"].probabilities   # {"delivery": 0.93, "billing": 0.05, "account": 0.02}
```

All the questions of one `system_one` call go to the model in a single batch. Make one call per state.

### What `from_pretrained` accepts

| `source` | example | loads |
|----------|---------|-------|
| a run directory | `runs/pointer/X` | its last finished stage, or `stage="ce"` |
| a stage directory | `runs/pointer/X/ce` | that stage (`adapter/` and `head.pt`) |
| a Hub repo id | `user/jev-pointer` | a stage uploaded with `jev-model push`; `revision=` picks a commit or tag |

The base model is downloaded by the name stored in `head.pt`, so a stage folder or repo holds only the LoRA and the
head (tens of MB). `device=""` picks CUDA when there is one.

### Loading from the Hugging Face Hub

A model on the Hub is a repo with one stage's `adapter/` (the LoRA) and `head.pt` (the pointer head, its settings
and the fitted temperature), as `jev-model push` uploads it. Pass the repo id:

```python
from jev_model.system_one import JevHolder

holder = JevHolder.from_pretrained("user/jev-pointer")
holder = JevHolder.from_pretrained("user/jev-pointer", revision="v1.0")   # a tag, branch or commit
holder = JevHolder.from_pretrained("user/jev-pointer", device="cpu")
```

What happens:

1. `source` is not a local run or stage folder, so it is taken as a repo id and downloaded with
   `huggingface_hub.snapshot_download` (at `revision`, if given).
2. `head.pt` names the base model (`unsloth/Qwen3.5-2B-Base` for the released runs), which is downloaded too
   (about 4 GB the first time).
3. The LoRA is applied to the base model and the head is loaded, as for a local run.

Both downloads go to the Hugging Face cache (`~/.cache/huggingface/hub`, or `$HF_HOME/hub`), so later loads are
local. Things to know:

- **Private repos:** log in once with `uv run hf auth login`, or set
  `HF_TOKEN` in the environment or in `.env`.
- **Offline machines:** download once while online, then set `HF_HUB_OFFLINE=1` to load from the cache only.
  You can also pass a local copy instead: `huggingface_hub.snapshot_download("user/jev-pointer", local_dir="models/jev")`,
  then `JevHolder.from_pretrained("models/jev")`. The base model still has to be in the cache.
- **Path or repo id:** a path that does not exist locally is treated as a repo id, so a typo in a local path shows
  up as a Hub "repository not found" error.
- **The CLI and the UI take runs only.** `jev-model predict`, `eval` and `ui` read a run directory; use
  `JevHolder` for a Hub model. To serve a Hub model with vLLM, see [Exporting a Hub model](vllm.md#from-a-hub-model).

To publish a run of your own:

```bash
uv run jev-model push runs/pointer/X --repo user/jev-pointer [--stage ce] [--private]
```

It creates the repo if needed and uploads only `adapter/` and `head.pt`.

### How questions become options

| question | options the model reads |
|----------|-------------------------|
| `Noul(...)` | `yes`, `no`, always in English and in this order, whatever the language of the instructions |
| `Choice(..., {"billing": "charges or invoices", "sales": None})` | `billing: charges or invoices`, `sales` |

A label is a dict key, so labels must be unique. Option texts that would repeat (two `None` labels can't, but two
descriptions can) are fine, since each option keeps its label. A `Choice` with fewer than two criteria, an empty
`questions` dict, or anything that is not a `Noul` or a `Choice` raises `ValueError` before the model runs.

### Lower level: raw options

`Predictor.predict` takes `(state, question, options)` tuples and returns probabilities in the given option order.
Use it when you build the option texts yourself, for example to replay rows of a built dataset:

```python
probs = holder.predictor.predict([
    ("Alice is taller than Bob. Bob is taller than Carol.", "Who is the shortest?", ["Alice", "Bob", "Carol"]),
])
# e.g. [[0.01, 0.02, 0.97]]
```

`notebooks/system_one.ipynb` walks through both levels, mixed questions, a row from a built dataset and the errors.

## From the command line

```bash
uv run jev-model predict runs/pointer/X \
  --state "I was charged twice for one order." \
  --question "Is this about billing?" --option yes --option no
# prints, for example:
# 0.981  yes
# 0.019  no
```

`--stage ce` picks a stage; the last finished one is the default.

## In the browser

```bash
uv sync --extra model --extra ui
uv run jev-model ui            # http://localhost:8501
```

Pick a run, then type a state, a question and options, or load a row from a built dataset to compare the prediction
with its label. The page works through `JevModel.predictor`, so the `typesafe` baseline shows up
there too.

## Writing good inputs

The released run was trained on 54 datasets (classification, NLU, reading comprehension, logic,
spatial reasoning, safety, ...) reshaped into `state -> question -> options`. It answers best when
your input looks like them.

- **Put everything the answer needs in the state.** The model is not meant to know facts. "Is Paris the capital of
  France?" with an empty state is a guess; the same question about a passage that says so is not. See
  [How the model works](model.md#what-it-is-for).
- **Ask one thing per question.** Several `Noul`s about one state are cheap: they run in one batch. A single
  question that bundles two conditions is harder for the model and for you to read.
- **Make options mutually exclusive and complete.** The probabilities always sum to 1 over the options you give.
  If none of them may fit, add an explicit option such as `none of these` (or `other`); the model was trained with
  a *None of the above* option in 10% of draws and uses it.
- **Describe labels that are not self-explanatory.** `{"p1": None, "p2": None}` gives the model nothing to go on;
  `{"p1": "outage affecting all users", "p2": "a single user is blocked"}` does. Short descriptions work better than
  long ones.
- **Keep the order stable.** Options attend to the options before them, so reordering them can shift the
  probabilities a little. Training shuffles the order to keep this small, but the output is not order-invariant.
- **Watch the length.** A prompt over `max_length` tokens (1024 in the released run) loses the *middle* of its
  state, replaced by an ellipsis; the question and options are always kept whole. If the question and options alone
  do not fit, the call raises `ValueError`. Split long documents yourself if the middle matters.
- **Read the probability, not just the argmax.** Each run fits a temperature on its validation data, so a 0.9
  means right about 90% of the time on data like the training mix (test ECE is about 0.04). Pick thresholds per
  question on a few labelled examples of your own.

## Scoring a run

```bash
uv run jev-model eval runs/pointer/X                       # test partition, last stage
uv run jev-model eval runs/pointer/X --partition validation --stage ce
```

`eval` prints accuracy, macro precision/recall/F1, NLL and ECE per dataset plus their `macro` mean, and saves them to
`RUN/<stage>-<partition>.json` (`last-<partition>.json` without `--stage`). `notebooks/compare_runs.ipynb` compares
saved scores across runs. The released run (`runs/pointer/20261003-174824`) scores a macro accuracy of 0.865, NLL
0.38 and ECE 0.039 over 54 test sets.
