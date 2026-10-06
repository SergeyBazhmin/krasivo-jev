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
  `with_context`, `disjoint_contexts`, `stratified_limit`.
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
|                  | `massive_intent_ru`           | choice   | `mteb/MassiveIntentClassification` (`ru`) |
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
|                  | `danetqa`                     | choice   | `RussianNLP/russian_super_glue` (`DaNetQA`) |
|                  | `lidirus`                     | choice   | `RussianNLP/russian_super_glue` (`LiDiRus`, evaluation only) |
|                  | `mera_multiq`                 | choice   | `ai-forever/MERA` (`multiq`, evidence selection) |
|                  | `multi_nli`                   | choice   | `nyu-mll/multi_nli` |
|                  | `multi_nli_genre`             | choice   | `nyu-mll/multi_nli` |
|                  | `muserc`                      | choice   | `RussianNLP/russian_super_glue` (`MuSeRC`) |
|                  | `paws`                        | noul     | `google-research-datasets/paws` (`labeled_final`) |
|                  | `race`                        | choice   | `ehovy/race` (`all`) |
|                  | `rcb`                         | choice   | `RussianNLP/russian_super_glue` (`RCB`) |
|                  | `ru_paraphraser`              | choice   | `merionum/ru_paraphraser` (three classes) |
|                  | `rublimp`                     | choice   | `RussianNLP/RuBLiMP` (45 tasks, evaluation only) |
|                  | `rucola`                      | choice   | `RussianNLP/rucola` |
|                  | `russe`                       | choice   | `RussianNLP/russian_super_glue` (`RUSSE`) |
|                  | `sberquad`                    | choice   | `kuznetsoffandrey/sberquad` (evidence selection) |
|                  | `terra`                       | choice   | `RussianNLP/russian_super_glue` (`TERRa`) |
| `reasoning`      | `logiqa`                      | choice   | `lucasmccabe/logiqa` |
|                  | `mera_bps`                    | choice   | `ai-forever/MERA` (`bps`) |
|                  | `mera_lcs`                    | choice   | `ai-forever/MERA` (`lcs`) |
|                  | `mera_mathlogicqa`            | choice   | `ai-forever/MERA` (`mathlogicqa`, math subset) |
|                  | `mera_rumodar`                | choice   | `ai-forever/MERA` (`rumodar`) |
|                  | `mera_rumultiar`              | choice   | `ai-forever/MERA` (`rumultiar`) |
|                  | `mera_simplear`               | choice   | `ai-forever/MERA` (`simplear`) |
|                  | `reclor`                      | choice   | `tasksource/reclor` |
| `retrieval`      | `nomiracl_en`                 | noul     | `miracl/nomiracl` (English, evaluation only) |
|                  | `nomiracl_ru`                 | choice   | `miracl/nomiracl` (Russian, evaluation only) |
|                  | `wikiqa`                      | choice   | `microsoft/wiki_qa` (passage selection) |
| `sentiment`      | `cedr`                        | choice   | `sagteam/cedr_v1` (`main`, independent emotion labels) |
|                  | `emotion`                     | choice   | `dair-ai/emotion` (`split`) |
|                  | `go_emotions`                 | choice   | `google-research-datasets/go_emotions` (`simplified`) |
|                  | `imdb`                        | choice   | `stanfordnlp/imdb` |
|                  | `sst2`                        | choice   | `stanfordnlp/sst2` |
|                  | `tweet_sentiment`             | choice   | `cardiffnlp/tweet_eval` (`sentiment`) |
| `spatial`        | `sparp`                       | choice   | `UKPLab/sparp` (`SpaRP-PS1 (SpaRTUN)`) |
|                  | `spartqa_mchoice`             | choice   | `tasksource/spartqa-mchoice` |
|                  | `spartqa_yn`                  | choice   | `tasksource/spartqa-yn` |
|                  | `stepgame`                    | choice   | `tasksource/stepgame` |
| `structured`     | `catalog`                     | choice   | deterministic synthetic catalogs (English) |
|                  | `catalog_ru`                  | choice   | deterministic synthetic catalogs (Russian) |
|                  | `cuad_spans`                  | choice   | official CUAD v1 contracts (literal fields only) |
|                  | `dialogsum_turns`             | choice   | official DialogSum transcripts (speaker/order targets) |
|                  | `e2e_slots`                   | choice   | cleaned E2E NLG descriptions and meaning representations |
|                  | `massive_slots`               | choice   | `AmazonScience/massive` (`en-US`, original slot annotations) |
|                  | `massive_slots_ru`            | choice   | `AmazonScience/massive` (`ru-RU`, original slot annotations) |
|                  | `multiwoz_state`              | choice   | official MultiWOZ 2.2 dialogues and schema |
|                  | `nerel_relations`             | choice   | official NEREL v1.0 documents and relation spans |
|                  | `sgd_state`                   | choice   | official Schema-Guided Dialogue dialogues and schemas |
|                  | `skillspan`                   | choice   | official SkillSpan job-posting excerpts and BIO spans |
|                  | `totto_cells`                 | choice   | official ToTTo tables (computed cell-selection targets) |
|                  | `webnlg_en`                   | choice   | `GEM/web_nlg` (`en`, facts and lexicalizations) |
|                  | `webnlg_ru`                   | choice   | `GEM/web_nlg` (`ru`, controlled renderings of supplied facts) |
|                  | `wikisql_queries`             | choice   | official WikiSQL tables and queries (computed English questions) |
|                  | `wikisql_queries_ru`          | choice   | WikiSQL tables with Russian question templates |
| `toxicity`       | `beavertails`                 | noul     | `PKU-Alignment/BeaverTails` |
|                  | `russian_toxicity`            | choice   | `textdetox/multilingual_toxicity_dataset` (`ru`) |
|                  | `toxic_chat`                  | noul     | `lmsys/toxic-chat` (`toxicchat0124`) |
|                  | `tweet_hate`                  | choice   | `cardiffnlp/tweet_eval` (`hate`) |
|                  | `tweet_offensive`             | choice   | `cardiffnlp/tweet_eval` (`offensive`) |
| `truthfulness`   | `halueval_dialogue`           | noul     | `pminervini/HaluEval` (`dialogue`) |
|                  | `halueval_qa`                 | noul     | `pminervini/HaluEval` (`qa`) |
|                  | `halueval_summarization`      | noul     | `pminervini/HaluEval` (`summarization`) |

Many logic and spatial sources bundle the same underlying sets (SpaRP includes StepGame, RuleCollection bundles
ProntoQA, LogicNLI, ProofWriter and more). Each set is taken from one source only, so test rows don't leak into
another dataset's train split.

Russian binary options read «да» / «нет» and use `choice`; `noul` requires literal English yes/no.
The new Russian converters remove shared contexts from lower-priority source splits, preserving test before
validation before train. RuParaPhraser additionally removes training pairs containing test headlines.

## Structured sources

These converters load official JSON/archives directly, so they do not depend on discontinued Hub loading scripts.
GitHub data revisions are pinned. Raw source groups are assigned to splits before producing multiple questions,
with test taking priority over validation and train for duplicate documents, dialogues or tables.
Raw record generation and the second cohort's transformations include converter/shared-helper code in their cache
keys, so edits to imported filters cannot silently reuse an older conversion.

- `massive_slots*`: extract one of 27 concrete slot types from the command, validating annotation markup against
  the original text. Alternative values of the same slot are sampled from train only; all accepted mentions are
  excluded. The original partitions are retained and repeated commands are removed across splits.
- `sgd_state`, `multiwoz_state`: track changed user belief values using dialogue prefixes and natural-language
  schema descriptions. Hidden service results, goal descriptions and gold belief states are never input.
  Values without visible evidence, unspecified/preference-free values and boolean availability slots are skipped.
  Equivalent common clock formats are excluded from distractors. At most four questions
  are produced per dialogue, with all its prefixes in one partition.
- `nerel_relations`: select literal object spans for a restricted set of temporal, family, workplace and name
  relations. Mentions must be separate spans in one sentence. Multiple distinct accepted objects, malformed spans,
  implicit geopolitical/medical/ideological relations and entity-linking annotations are excluded. Alias and
  nested mentions of the accepted object are excluded from distractors.
- `totto_cells`: read a value by a unique row key and a column header. Only small rectangular tables with an
  unambiguous first header row and no merged cells are kept. Distractors come from the same column. Original
  summaries and highlights are excluded; labels are computed from cells, including public test tables. Shared
  tables are removed from training regardless of different original highlight selections.
- `webnlg_*`: compare a specified relation's value against descriptions with that value replaced; the rest of the
  sentence is not judged. English uses original lexicalizations with literal subject/object matches. Russian uses
  explicitly synthetic renderings of facts with untranslated entity strings, because the original Russian text
  translates names without providing their alignment. Replacements exclude every accepted object for the same
  subject/relation. Alternatives are taken from that relation and a train-only bank; numeric alternatives are
  numeric. All references to a fact set share a split.
- `catalog*`: synthetic catalogs support conjunctive filtering, cheapest eligible item selection, and inventory
  updates. Prices are unique, ambiguous filter questions are skipped, and every label is calculated in Python.
  English/Russian versions share generation keys and split assignments.
- `skillspan`: use literal BIO spans from both skill/knowledge layers to choose a phrase explicitly present in a
  job-posting excerpt. This does not predict ESCO codes, implied requirements or hard/soft taxonomy labels.
  Every negative must be absent from the entire excerpt, so incomplete annotations cannot create false negatives.
  Source job ids restart in each partition; job content groups all its sentences before conversion, and duplicate
  excerpts are removed across partitions. One question is produced for each eligible sentence.
- `e2e_slots`: use the fully cleaned, non-overlapping E2E release. Restaurant names, nearby landmarks, cuisine,
  establishment types and areas are kept only when their values appear literally in the human description.
  Attribute mentions inside restaurant/landmark names do not count as independent evidence. Ratings, family
  suitability and normalized prices are excluded. The meaning representation supplies targets but is never input;
  its references remain in one source partition, and repeated descriptions are removed across partitions.
- `dialogsum_turns`: infer the author of a quoted utterance from speaker tags, or find the next recorded utterance
  from transcript order. The task asks what actually happened, not which reply sounds plausible. Repeated quotes
  with ambiguous authors/successors and malformed transcripts are skipped. Original summaries/topics are unused.
- `cuad_spans`: select original wording for contract titles, agreement dates, explicitly stated effective dates and
  governing law. Exact annotation offsets must match. Dates must be literal dates with a year; effective dates
  require a nearby explicit effective-date cue, rather than assuming that signing makes a contract effective.
  Relative dates, multi-clause reasoning, legal-effect labels and absent-answer annotations are omitted. Context is
  a short unmarked window around a span; distractors come from that field's training bank and cannot occur in the
  passage or match another accepted answer. The source test contracts are retained; validation is carved from
  source train by a contract-content hash before windows are made.
- `wikisql_queries*`: generate precise natural-language questions from the query's selection, conjunction of
  comparisons and aggregation, then execute them over the public table using Decimal arithmetic. Operations are
  lookup, maximum/minimum, row count, sum and mean rounded to two decimal places. No original natural-language
  question or SQL text enters the sample. Unsupported numeric formats/units, text inequalities, ambiguous
  list-valued answers, large/ragged tables and duplicate column headers are excluded. Equivalent numeric
  distractors are removed. All questions about a table share a partition; canonical fingerprints ignore row order
  and source table ids. Russian templates retain original table headers/cells, and preserve the English groups,
  labels and options.

WikiSQL and ToTTo contain some of the same Wikipedia tables in conflicting splits. The checked-in
`structured/wikisql_totto_overlap.json` records their common canonical table hashes and archive checksums; every
such table is excluded from WikiSQL in all partitions, leaving it to ToTTo. Regenerate the manifest when updating
the WikiSQL source revision with `scripts/audit_table_overlap.py --wikisql ARCHIVE --totto ARCHIVE --output MANIFEST`.
This audit detects complete-table duplicates, not arbitrary overlapping subsets of rows or semantic paraphrases.

Source attribution and terms: [SkillSpan](https://github.com/kris927b/SkillSpan) (repository MIT),
[E2E](https://github.com/tuetschek/e2e-dataset) (CC BY-SA 4.0) with the
[authors' cleaned release](https://github.com/tuetschek/e2e-cleaning),
[DialogSum](https://github.com/cylnlp/dialogsum) (**CC BY-NC-SA 4.0**, noncommercial),
[CUAD](https://www.atticusprojectai.org/cuad/) (CC BY 4.0), and
[WikiSQL](https://github.com/salesforce/WikiSQL/blob/master/LICENSE) (BSD 3-Clause repository license).
Conversion does not replace the original source terms.

```bash
uv run jev prepare massive_slots massive_slots_ru sgd_state multiwoz_state nerel_relations \
  totto_cells webnlg_en webnlg_ru catalog catalog_ru --max-samples 10000
uv run jev prepare skillspan e2e_slots dialogsum_turns cuad_spans \
  wikisql_queries wikisql_queries_ru --max-samples 10000
```

Offline semantic checks for these two cohorts are in `scripts/smoke_structured.py` and
`scripts/smoke_more_structured.py`. Each accepts `--built-dir` for checking every saved sample and disjoint contexts.
The second script also checks the table interpreter against independently executed SQLite queries, effective-date
cues, malformed spans, repeated dialogue utterances and the alignment of the two WikiSQL template languages.

## Adding a dataset

See `AGENTS.md`. In short: check the source is answerable from the state alone, subclass `JevDataset`, map rows
with `features=SAMPLE_FEATURES`, give it ~5 question paraphrases, shuffle options deterministically when position
gives the answer away, and register it in `__init__.py`. `sentiment/sst2.py` is the simplest example. Check it with
`uv run jev prepare <name> --max-samples 50 --output-dir <scratch>`.
