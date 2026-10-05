# jev_datasets

Converts Hugging Face datasets into one multiple-choice format, `state -> question -> options -> label`, for a model
that answers in one forward pass from the given context alone. A source, subset or row that needs world or expert
knowledge beyond the `state` is left out: the model is not meant to store facts.

## Sample

| field      | type                          | meaning |
|------------|-------------------------------|---------|
| `state`    | `str`                         | the context the question is about (`""` if there is none) |
| `question` | `str`                         | the question, one of ~5 paraphrases picked per sample |
| `options`  | `[{"id": str, "text": str}]`  | answer candidates: `text` is what the model reads, `id` a stable key |
| `label`    | `[float32]`                   | probability distribution over `options` (one-hot, or soft such as vote shares) |
| `type`     | `str`                         | `noul` (the options are only yes/no) or `choice` |

The schema is `SAMPLE_FEATURES` in `utils.py`; `make_sample` builds a row from the id of the correct option or a
distribution.

## Usage

```bash
uv run jev list                                              # registered datasets
uv run jev prepare sst2 anli                                 # some of them, into data/<name>
uv run jev prepare all --max-samples 5000 --output-dir /path/to/out
```

`prepare` loads the source from the Hub, converts it, stamps `type` on every row, caps every split at
`--max-samples` (10,000 by default, stratified by the argmax of `label`) and saves a `DatasetDict` to
`OUTPUT_DIR/<name>`. Read it back with `datasets.load_from_disk`.

Splits are kept as the source has them; `jev_model.data.partition` carves a missing validation or test split from
train by a content hash, so every model is scored on the same rows.

## Files

- `base.py`: `JevDataset(name, hf_path, hf_name)` loads the source lazily into `.data`; a subclass sets `type` and
  implements `prepare()`, and overrides `load()` when the source needs special loading.
- `utils.py`: `make_sample`, `make_options`, `pick_question`, `stable_shuffle`, `explode`, `slugify`,
  `with_context`, `stratified_limit`.
- `constants.py`: `MAX_SAMPLES` and `ROOT_DIR` (`data/` at the repo root).
- `cli.py`: the `jev` CLI.
- `__init__.py`: the `datasets` registry, `{name: instance}`, alphabetical.

## Datasets

| topic            | dataset                       | type     | source |
|------------------|-------------------------------|----------|--------|
| `bugs`           | `github_issue_type`           | choice   | `khanmu2003/issue-classification` |
|                  | `vscode_triage`               | choice   | `Tilakoid/vscode-bug-feature-triage` |
| `business`       | `financial_news_topic`        | choice   | `zeroshot/twitter-financial-news-topic` |
| `classification` | `ag_news`                     | choice   | `fancyzhx/ag_news` |
|                  | `atis`                        | choice   | `tuetschek/atis` |
|                  | `banking77`                   | choice   | `legacy-datasets/banking77` |
|                  | `bitext_customer_support`     | choice   | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` |
|                  | `bitext_insurance`            | choice   | `bitext/Bitext-insurance-llm-chatbot-training-dataset` |
|                  | `bitext_retail_banking`       | choice   | `bitext/Bitext-retail-banking-llm-chatbot-training-dataset` |
|                  | `bitext_telco`                | choice   | `bitext/Bitext-telco-llm-chatbot-training-dataset` |
|                  | `clinc_oos`                   | choice   | `clinc/clinc_oos` (`plus`) |
|                  | `dbpedia`                     | choice   | `fancyzhx/dbpedia_14` |
|                  | `hwu64`                       | choice   | `DeepPavlov/hwu64` |
|                  | `massive_intent`              | choice   | `mteb/amazon_massive_intent` (`en`) |
|                  | `ticket_routing`              | choice   | `Tobi-Bueck/customer-support-tickets` |
|                  | `ticket_type`                 | choice   | `Tobi-Bueck/customer-support-tickets` |
| `guardrails`     | `aegis_safety`                | noul     | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` |
|                  | `jailbreak_classification`    | noul     | `jackhhao/jailbreak-classification` |
|                  | `safeguard_prompt_injection`  | noul     | `xTRam1/safe-guard-prompt-injection` |
|                  | `spml_prompt_injection`       | noul     | `reshabhs/SPML_Chatbot_Prompt_Injection` |
| `knowledge`      | `sciq`                        | choice   | `allenai/sciq` (only rows with a support passage) |
| `logic`          | `ar_lsat`                     | choice   | `RuleReasoner/RuleCollection-32K` |
|                  | `clutrr`                      | choice   | `tasksource/clutrr` |
|                  | `folio`                       | choice   | `yfxiao/folio-refined` |
|                  | `logicnli`                    | choice   | `tasksource/LogicNLI` |
|                  | `prontoqa`                    | choice   | `RuleReasoner/RuleCollection-32K` |
|                  | `proofwriter`                 | choice   | `tasksource/proofwriter` |
|                  | `proverqa`                    | choice   | `opendatalab/ProverQA` |
| `nlu`            | `anli`                        | choice   | `facebook/anli` |
|                  | `babi_nli`                    | noul     | `tasksource/babi_nli` |
|                  | `boolq`                       | noul     | `google/boolq` |
|                  | `chaos_nli`                   | choice   | `tasksource/chaos-mnli-ambiguity` (soft labels) |
|                  | `multi_nli`                   | choice   | `nyu-mll/multi_nli` |
|                  | `multi_nli_genre`             | choice   | `nyu-mll/multi_nli` |
|                  | `paws`                        | noul     | `google-research-datasets/paws` (`labeled_final`) |
|                  | `race`                        | choice   | `ehovy/race` (`all`) |
| `reasoning`      | `logiqa`                      | choice   | `lucasmccabe/logiqa` |
|                  | `reclor`                      | choice   | `tasksource/reclor` |
| `sentiment`      | `emotion`                     | choice   | `dair-ai/emotion` (`split`) |
|                  | `go_emotions`                 | choice   | `google-research-datasets/go_emotions` (`simplified`) |
|                  | `imdb`                        | choice   | `stanfordnlp/imdb` |
|                  | `sst2`                        | choice   | `stanfordnlp/sst2` |
|                  | `tweet_sentiment`             | choice   | `cardiffnlp/tweet_eval` (`sentiment`) |
| `spatial`        | `sparp`                       | choice   | `UKPLab/sparp` (`SpaRP-PS1 (SpaRTUN)`) |
|                  | `spartqa_mchoice`             | choice   | `tasksource/spartqa-mchoice` |
|                  | `spartqa_yn`                  | choice   | `tasksource/spartqa-yn` |
|                  | `stepgame`                    | choice   | `tasksource/stepgame` |
| `toxicity`       | `beavertails`                 | noul     | `PKU-Alignment/BeaverTails` |
|                  | `toxic_chat`                  | noul     | `lmsys/toxic-chat` (`toxicchat0124`) |
|                  | `tweet_hate`                  | choice   | `cardiffnlp/tweet_eval` (`hate`) |
|                  | `tweet_offensive`             | choice   | `cardiffnlp/tweet_eval` (`offensive`) |
| `truthfulness`   | `halueval_dialogue`           | noul     | `pminervini/HaluEval` (`dialogue`) |
|                  | `halueval_qa`                 | noul     | `pminervini/HaluEval` (`qa`) |
|                  | `halueval_summarization`      | noul     | `pminervini/HaluEval` (`summarization`) |

Many logic and spatial sources bundle the same underlying sets (SpaRP includes StepGame, RuleCollection bundles
ProntoQA, LogicNLI, ProofWriter and more). Each set is taken from one source only, so test rows don't leak into
another dataset's train split.

## Adding a dataset

See `AGENTS.md`. In short: check the source is answerable from the state alone, subclass `JevDataset`, map rows
with `features=SAMPLE_FEATURES`, give it ~5 question paraphrases, shuffle options deterministically when position
gives the answer away, and register it in `__init__.py`. `sentiment/sst2.py` is the simplest example. Check it with
`uv run jev prepare <name> --max-samples 50 --output-dir <scratch>`.
