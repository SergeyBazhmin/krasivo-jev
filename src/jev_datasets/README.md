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
uv run jev prepare all -j 8                                  # 8 datasets at once, one process each
```

`prepare` loads the source from the Hub, converts it, regroups the splits (below), stamps `type` on every row, caps
every split at `--max-samples` (10,000 by default, a seeded random sample) and saves a `DatasetDict` to
`OUTPUT_DIR/<name>`. Read it back with `datasets.load_from_disk`.

Every saved dataset has exactly the splits `train`, `validation` and `test`. Source splits are renamed (`dev`/`val` to
`validation`; unknown names such as halueval's `data` join `train`). Rows sharing a context are grouped, and each group
is moved whole into the first split it appears in, held-out splits first (test, validation, train). The group is the
row's `group` column when the converter emits one (a source document that yields several states), else a hash of
`state`, or of `question` when there is no state. No context is in two splits. Exact duplicate rows are dropped,
keeping the held-out copy, and no other row is lost. Splits the source lacks are then carved by a hash of the group:
5% of train each for a missing validation or test, or, when there is no train split (e.g. `rublimp`), all rows are
re-split 80/10/10. Carving happens before the cap, so every model is scored on the same rows.

## Files

- `base.py`: `JevDataset(name, hf_path, hf_name)` loads the source lazily into `.data`; a subclass sets `type` and
  implements `prepare()`, and overrides `load()` when the source needs special loading.
- `utils.py`: `make_sample`, `make_options`, `pick_question`, `pick_about`, `stable_shuffle`, `explode`, `slugify`,
  `with_context`, `GROUPED_FEATURES`, `disjoint_contexts`, `random_limit`.
- `constants.py`: `MAX_SAMPLES` and `ROOT_DIR` (`data/` at the repo root).
- `cli.py`: the `jev` CLI.
- `__init__.py`: the `datasets` registry, `{name: instance}`, alphabetical.

## Datasets

| topic            | dataset                       | type     | source |
|------------------|-------------------------------|----------|--------|
| `bugs`           | `github_issue_type`           | choice   | `khanmu2003/issue-classification` |
|                  | `vscode_triage`               | choice   | `Tilakoid/vscode-bug-feature-triage` |
| `business`       | `financial_news_topic`        | choice   | `zeroshot/twitter-financial-news-topic` |
|                  | `open_decisions_choice`       | choice   | `jgeuter/OpenDecisions` (choice items, as published, benchmark) |
|                  | `open_decisions_noul`         | noul     | `jgeuter/OpenDecisions` (yes/no items, as published, benchmark) |
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
|                  | `rus_med_dialogues`           | choice   | `Mykes/rus_med_dialogues` (patient question -> specialist) |
|                  | `rus_med_dialogues_noul`      | noul     | `Mykes/rus_med_dialogues` (patient question -> specialist) |
|                  | `ticket_routing`              | choice   | `Tobi-Bueck/customer-support-tickets` |
|                  | `ticket_type`                 | choice   | `Tobi-Bueck/customer-support-tickets` |
| `guardrails`     | `aegis_category`              | choice   | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` (unsafe prompts, soft labels) |
|                  | `jailbreak_classification`    | noul     | `jackhhao/jailbreak-classification` |
|                  | `safeguard_prompt_injection`  | noul     | `xTRam1/safe-guard-prompt-injection` |
|                  | `spml_prompt_injection`       | noul     | `reshabhs/SPML_Chatbot_Prompt_Injection` |
| `knowledge`      | `qasc`                        | choice   | `allenai/qasc` (the two facts as state; unlabelled test dropped) |
|                  | `quartz`                      | choice   | `allenai/quartz` |
|                  | `ropes`                       | choice   | `allenai/ropes` (`plain_text`, other answers on the same situation as distractors) |
|                  | `sciq`                        | choice   | `allenai/sciq` (only rows with a support passage) |
| `logic`          | `ar_lsat`                     | choice   | `RuleReasoner/RuleCollection-32K` |
|                  | `clutrr`                      | choice   | `tasksource/clutrr` |
|                  | `folio`                       | choice   | `yfxiao/folio-refined` |
|                  | `logicnli`                    | choice   | `tasksource/LogicNLI` |
|                  | `proofwriter`                 | choice   | `tasksource/proofwriter` |
|                  | `proverqa`                    | choice   | `opendatalab/ProverQA` |
| `nlu`            | `anli`                        | choice   | `facebook/anli` |
|                  | `babi_nli`                    | noul     | `tasksource/babi_nli` |
|                  | `boolq`                       | noul     | `google/boolq` |
|                  | `chaos_nli`                   | choice   | `tasksource/chaos-mnli-ambiguity` (soft labels) |
|                  | `danetqa`                     | choice   | `RussianNLP/russian_super_glue` (`DaNetQA`) |
|                  | `dream`                       | choice   | `dataset-org/dream` (parquet export) |
|                  | `hotpot_qa`                   | choice   | `hotpotqa/hotpot_qa` (`distractor`, comparison questions answered by an entity) |
|                  | `hotpot_qa_noul`              | noul     | `hotpotqa/hotpot_qa` (`distractor`, comparison questions answered yes/no) |
|                  | `lidirus`                     | choice   | `RussianNLP/russian_super_glue` (`LiDiRus`, evaluation only) |
|                  | `mera_multiq`                 | choice   | `ai-forever/MERA` (`multiq`, evidence selection) |
|                  | `multi_nli`                   | choice   | `nyu-mll/multi_nli` |
|                  | `multi_nli_genre`             | choice   | `nyu-mll/multi_nli` |
|                  | `muserc`                      | choice   | `RussianNLP/russian_super_glue` (`MuSeRC`) |
|                  | `paws`                        | noul     | `google-research-datasets/paws` (`labeled_final`) |
|                  | `quail`                       | choice   | `textmachinelab/quail` (without `Event_duration` questions) |
|                  | `race`                        | choice   | `ehovy/race` (`all`) |
|                  | `rcb`                         | choice   | `RussianNLP/russian_super_glue` (`RCB`) |
|                  | `ru_paraphraser`              | choice   | `merionum/ru_paraphraser` (three classes) |
|                  | `rublimp`                     | choice   | `RussianNLP/RuBLiMP` (45 tasks, evaluation only) |
|                  | `rucola`                      | choice   | `RussianNLP/rucola` |
|                  | `russe`                       | choice   | `RussianNLP/russian_super_glue` (`RUSSE`) |
|                  | `sberquad`                    | choice   | `kuznetsoffandrey/sberquad` (evidence selection) |
|                  | `terra`                       | choice   | `RussianNLP/russian_super_glue` (`TERRa`) |
| `reasoning`      | `logiqa`                      | choice   | `lucasmccabe/logiqa` |
|                  | `reclor`                      | choice   | `tasksource/reclor` |
| `retrieval`      | `nomiracl_choice_en`          | choice   | `miracl/nomiracl` (English, which passage is relevant or none, benchmark) |
|                  | `nomiracl_choice_ru`          | choice   | `miracl/nomiracl` (Russian, which passage is relevant or none, benchmark) |
|                  | `nomiracl_en`                 | noul     | `miracl/nomiracl` (English, does any passage answer the query, benchmark) |
|                  | `nomiracl_passage_en`         | noul     | `miracl/nomiracl` (English, is one passage relevant, benchmark) |
|                  | `nomiracl_passage_ru`         | noul     | `miracl/nomiracl` (Russian, is one passage relevant, benchmark) |
|                  | `nomiracl_ru`                 | noul     | `miracl/nomiracl` (Russian, does any passage answer the query, benchmark) |
|                  | `ticket_subject`              | choice   | `Tobi-Bueck/customer-support-tickets` (subject → email body) |
|                  | `wikiqa`                      | choice   | `microsoft/wiki_qa` (passage selection) |
|                  | `wikiqa_noul`                 | noul     | `microsoft/wiki_qa` (does a passage answer the query) |
| `sentiment`      | `cedr`                        | choice   | `sagteam/cedr_v1` (`main`, independent emotion labels) |
|                  | `emotion`                     | choice   | `dair-ai/emotion` (`split`) |
|                  | `finentity`                   | choice   | `yixuantt/FinEntity` (sentiment toward each entity) |
|                  | `go_emotions`                 | choice   | `google-research-datasets/go_emotions` (`simplified`) |
|                  | `go_emotions_noul`            | noul     | `google-research-datasets/go_emotions` (`simplified`, one row per emotion) |
|                  | `imdb`                        | choice   | `stanfordnlp/imdb` |
|                  | `sst2`                        | choice   | `stanfordnlp/sst2` |
|                  | `tweet_sentiment`             | choice   | `cardiffnlp/tweet_eval` (`sentiment`) |
| `spatial`        | `sparp`                       | choice   | `UKPLab/sparp` (`SpaRP-PS1 (SpaRTUN)`) |
|                  | `spartqa_mchoice`             | choice   | `tasksource/spartqa-mchoice` |
|                  | `spartqa_yn`                  | choice   | `tasksource/spartqa-yn` |
|                  | `stepgame`                    | choice   | `tasksource/stepgame` |
| `structured`     | `atis_slots`                  | choice   | `tuetschek/atis` (slot spans, distractors from the same request) |
|                  | `cuad_spans`                  | choice   | official CUAD v1 contracts (literal fields only) |
|                  | `massive_slots`               | choice   | `AmazonScience/massive` (`en-US`, original slot annotations) |
|                  | `massive_slots_ru`            | choice   | `AmazonScience/massive` (`ru-RU`, original slot annotations) |
|                  | `multiwoz_state`              | choice   | official MultiWOZ 2.2 dialogues and schema |
|                  | `nerel_relations`             | choice   | official NEREL v1.0 documents and relation spans |
|                  | `sgd_state`                   | choice   | official Schema-Guided Dialogue dialogues and schemas |
|                  | `skillspan`                   | choice   | official SkillSpan job-posting excerpts and BIO spans |
|                  | `totto_cells`                 | choice   | official ToTTo tables (computed cell-selection targets) |
|                  | `totto_statements`            | choice   | official ToTTo tables and annotator sentences (other tables as distractors) |
|                  | `webnlg_en`                   | choice   | `GEM/web_nlg` (`en`, facts and lexicalizations) |
| `tool`           | `api_bank`                    | choice   | `liminghao1630/API-Bank` (level 1-2 API requests, distractors from the same dialogue) |
|                  | `bfcl`                        | choice   | `BitAgent/bfcl_shuffle_full` (function selection or none; not live simple/multiple) |
|                  | `when2call`                   | choice   | `nvidia/When2Call` (preference pairs and the 4-way MCQ, both in train; validation/test carved) |
| `toxicity`       | `beavertails`                 | noul     | `PKU-Alignment/BeaverTails` |
|                  | `nsfw_benchmark`              | choice   | `redmadrobot-rnd/nsfw_benchmark` (Russian and English) |
|                  | `russian_toxicity`            | choice   | `textdetox/multilingual_toxicity_dataset` (`ru`) |
|                  | `toxic_chat`                  | noul     | `lmsys/toxic-chat` (`toxicchat0124`) |
|                  | `tweet_hate`                  | choice   | `cardiffnlp/tweet_eval` (`hate`) |
|                  | `tweet_offensive`             | choice   | `cardiffnlp/tweet_eval` (`offensive`) |
| `truthfulness`   | `halueval_dialogue`           | noul     | `pminervini/HaluEval` (`dialogue`) |
|                  | `halueval_qa`                 | noul     | `pminervini/HaluEval` (`qa`) |
|                  | `halueval_summarization`      | noul     | `pminervini/HaluEval` (`summarization`) |
|                  | `llm_aggrefact`               | noul     | `lytang/LLM-AggreFact` (gated; includes RAGTruth) |
|                  | `vitaminc`                    | choice   | `tals/vitaminc` (real Wikipedia revisions only) |

Many logic and spatial sources bundle the same underlying sets (SpaRP includes StepGame, RuleCollection bundles
ProntoQA, LogicNLI, ProofWriter and more). Each set is taken from one source only, so test rows don't leak into
another dataset's train split.

`ticket_routing`, `ticket_type` and `ticket_subject` split the helpdesk tickets once, by a hash of the email body, so a
ticket lands in the same partition in all three.

Russian binary options read «да» / «нет» and use `choice`; `noul` requires literal English yes/no ids. NoMIRACL's
Russian yes/no sets are the exception: their ids are `yes`/`no` with «да» / «нет» as text, so they are `noul`.
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
- `atis_slots`: pick the words of an ATIS request that fill a concrete slot (departure/destination city, dates,
  times, airline, class, ...). Distractors are the request's other annotated spans, so the slot's role decides the
  answer rather than string matching. Modifiers, undirected cities and ontology codes are excluded, as are slots
  filled with two different values. Requests shared by train and test stay in test.
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
- `totto_statements`: pick the annotators' final sentence about a table among sentences written for other tables of
  the same split, preferring tables with the same headers so the cell values decide. Sentences about the same table
  or Wikipedia page are never distractors. Page and section titles are in the state; tables are filtered as in
  `totto_cells` and assigned the same splits. The unlabeled test split is not used.
- `webnlg_en`: compare a specified relation's value against descriptions with that value replaced; the rest of the
  sentence is not judged. Original lexicalizations are used with literal subject/object matches. The Russian
  release is not used: it translates names without providing their alignment. Replacements exclude every accepted
  object for the same subject/relation. Alternatives are taken from that relation and a train-only bank; numeric alternatives are
  numeric. All references to a fact set share a split.
- `skillspan`: use literal BIO spans from both skill/knowledge layers to choose a phrase explicitly present in a
  job-posting excerpt. This does not predict ESCO codes, implied requirements or hard/soft taxonomy labels.
  Every negative must be absent from the entire excerpt, so incomplete annotations cannot create false negatives.
  Source job ids restart in each partition; job content groups all its sentences before conversion, and duplicate
  excerpts are removed across partitions. One question is produced for each eligible sentence.
- `cuad_spans`: select original wording for contract titles, agreement dates, explicitly stated effective dates and
  governing law. Exact annotation offsets must match. Dates must be literal dates with a year; effective dates
  require a nearby explicit effective-date cue, rather than assuming that signing makes a contract effective.
  Relative dates, multi-clause reasoning, legal-effect labels and absent-answer annotations are omitted. Context is
  a short unmarked window around a span; distractors come from that field's training bank and cannot occur in the
  passage or match another accepted answer. The source test contracts are retained; validation is carved from
  source train by a contract-content hash before windows are made.

Source attribution and terms: [SkillSpan](https://github.com/kris927b/SkillSpan) (repository MIT) and
[CUAD](https://www.atticusprojectai.org/cuad/) (CC BY 4.0).
Conversion does not replace the original source terms.

```bash
uv run jev prepare massive_slots massive_slots_ru sgd_state multiwoz_state nerel_relations \
  totto_cells webnlg_en --max-samples 10000
uv run jev prepare skillspan cuad_spans --max-samples 10000
```

Offline semantic checks for these two cohorts are in `scripts/smoke_structured.py` and
`scripts/smoke_more_structured.py`. Each accepts `--built-dir` for checking every saved sample and disjoint contexts.
The second script also checks effective-date cues and malformed spans.

## Adding a dataset

See `AGENTS.md`. In short: check the source is answerable from the state alone, subclass `JevDataset`, map rows
with `features=SAMPLE_FEATURES`, give it ~5 question paraphrases, shuffle options deterministically when position
gives the answer away, and register it in `__init__.py`. `sentiment/sst2.py` is the simplest example. Check it with
`uv run jev prepare <name> --max-samples 50 --output-dir <scratch>`.
