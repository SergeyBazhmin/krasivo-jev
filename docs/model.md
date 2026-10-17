# How the model works

## What it is for

A "system one" model: it reads a text and answers a multiple-choice question about it in a single forward pass,
the way you'd answer at a glance. It is built to *read*, not to *remember*. The training data was filtered so that
every answer follows from the state and the question alone; sources that need world, cultural or expert knowledge
(MMLU, ARC, CommonsenseQA, ...) were removed on purpose.

Good fits:

- classifying text: intent, topic, sentiment, toxicity, routing a ticket to a team
- yes/no checks on a document: does it mention X, does it ask for Y, is it spam
- reading comprehension, entailment and contradiction between sentences
- small logic and spatial puzzles stated in the text
- many cheap questions about one text, each with a calibrated probability

Poor fits:

- facts that are not in the state (it will answer, but it is guessing)
- long multi-step reasoning or arithmetic over many steps
- open-ended answers: it only picks among the options you give
- states much longer than `max_length` (1024 tokens in the released run), whose middle is cut

## Architecture

```
<state> state text <q> question <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
                                                  │                    │           │
                                             hidden h₁            hidden h₂     hidden d
                                                  └──── logitᵢ = key(hᵢ) · query(d) · scale / T ────┘
```

- **Backbone:** `unsloth/Qwen3.5-2B-Base`, a hybrid decoder whose layers are mostly linear attention (Gated
  DeltaNet). It stays frozen; a LoRA (rank 16, on attention and MLP projections) adapts it.
- **Delimiters:** five rarely used Qwen special tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`,
  `<|box_end|>`, `<|fim_suffix|>`) mark the parts, so no embedding rows are added. A `<|name|>` in your own text is
  rewritten before tokenizing, so user text cannot forge an option boundary.
- **Pointer head:** projects the hidden state at each `</opt>` (key) and at `<decide>` (query) to 256 dimensions; an
  option's logit is their scaled dot product. There are no fixed output slots, so any number of options works and
  the option's own text drives its score.
- **Calibration:** one temperature `T`, fitted on the validation partition after training, divides the logits.
- **Attention:** plain causal, one question per sequence. Each option sees the state, the question and the options
  before it.

Because attention is causal and the options come last, the cost of a question is about the length of the state
plus its options; ten options are barely dearer than two.

## Training, briefly

- Data: source datasets (98 registered today; the released run used the 54 built at the time) converted to `state -> question -> options -> label` by `jev_datasets`,
  capped per split, with train/validation/test disjoint by context. Every model in the repo is scored on the same
  test rows.
- Loss: soft cross-entropy against the label distribution (focal loss is opt-in).
- Each draw of a sample shows its options in a fresh order, so the model does not learn a position prior.
- 10% of draws replace the true option's text with *None of the above* (which then carries the label), so the model
  learns that the listed options can all be wrong.
- 20% of draws drop some wrong options, so the number of options does not give the dataset away.
- Questions and instructions come in several paraphrases per dataset, so the model learns tasks, not prompts.
- The weights with the best macro NLL on validation are kept.

## Properties to rely on, and not to

| | |
|---|---|
| **Probabilities sum to 1 over your options** | It ranks the options you give. Add `none of these` when all may be wrong. |
| **Calibrated on the training mix** | Test ECE is about 0.04. On data unlike the mix, check calibration on a labelled sample before trusting thresholds. |
| **Not order-invariant** | Options attend to earlier ones. Shuffled training keeps the effect small; keep your order fixed for reproducibility. |
| **Deterministic** | Same input and weights give the same output, up to GPU numeric noise. |
| **Language** | Trained on English and Russian data. `Noul` options are always `yes`/`no`, even for Russian instructions. |
| **Length** | Over `max_length`, the middle of the state is replaced by an ellipsis; the question and options always stay whole. |
| **No knowledge** | Do not use it as a fact store. Put the facts in the state. |

## Other models in the repo

`jev-model` also wraps two baselines, scored on the same test rows so they compare directly with `pointer`:

- `zero_shot`: a pretrained chat model behind an OpenAI-compatible server (for example vLLM serving Gemma or Qwen,
  `src/scripts/serve_gemma.sh`, `serve_qwen.sh`) that replies with an option number.
- `typesafe`: TypeSafe's hosted `jev` model through its `system_one` API (needs `TYPESAFE_API_KEY` in `.env`).
  `JevHolder` and the vLLM endpoint use the same request and answer shape, so code written against one moves to the
  other.

Neither trains anything; `train` only records the config, and `eval` scores them.
