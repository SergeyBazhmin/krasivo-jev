---
base_model: Qwen/Qwen3.5-2B-Base
library_name: peft
pipeline_tag: text-classification
language:
- en
- ru
tags:
- multiple-choice
- zero-shot-classification
- pointer-network
- lora
- vllm
datasets:
- AmazonScience/massive
- DeepPavlov/hwu64
- GEM/totto
- GEM/web_nlg
- Mykes/rus_med_dialogues
- PKU-Alignment/BeaverTails
- RuleReasoner/RuleCollection-32K
- RussianNLP/RuBLiMP
- RussianNLP/rucola
- RussianNLP/russian_super_glue
- The-Atticus-Project/cuad
- Tilakoid/vscode-bug-feature-triage
- Tobi-Bueck/customer-support-tickets
- UKPLab/sparp
- ai-forever/MERA
- allenai/qasc
- allenai/quartz
- allenai/ropes
- allenai/sciq
- bitext/Bitext-customer-support-llm-chatbot-training-dataset
- bitext/Bitext-insurance-llm-chatbot-training-dataset
- bitext/Bitext-retail-banking-llm-chatbot-training-dataset
- bitext/Bitext-telco-llm-chatbot-training-dataset
- budzianowski/multiwoz
- cardiffnlp/tweet_eval
- clinc/clinc_oos
- dair-ai/emotion
- dataset-org/dream
- ehovy/race
- facebook/anli
- fancyzhx/ag_news
- fancyzhx/dbpedia_14
- google-research-datasets/dstc8-schema-guided-dialogue
- google-research-datasets/go_emotions
- google-research-datasets/paws
- google/boolq
- hotpotqa/hotpot_qa
- jackhhao/jailbreak-classification
- jgeuter/OpenDecisions
- khanmu2003/issue-classification
- kris927b/SkillSpan
- kuznetsoffandrey/sberquad
- legacy-datasets/banking77
- lmsys/toxic-chat
- lucasmccabe/logiqa
- merionum/ru_paraphraser
- microsoft/wiki_qa
- miracl/nomiracl
- mteb/MassiveIntentClassification
- mteb/amazon_massive_intent
- nerel-ds/NEREL
- nvidia/Aegis-AI-Content-Safety-Dataset-2.0
- nyu-mll/multi_nli
- opendatalab/ProverQA
- pminervini/HaluEval
- reshabhs/SPML_Chatbot_Prompt_Injection
- sagteam/cedr_v1
- stanfordnlp/imdb
- stanfordnlp/sst2
- tals/vitaminc
- tasksource/LogicNLI
- tasksource/babi_nli
- tasksource/chaos-mnli-ambiguity
- tasksource/clutrr
- tasksource/proofwriter
- tasksource/reclor
- tasksource/spartqa-mchoice
- tasksource/spartqa-yn
- tasksource/stepgame
- textdetox/multilingual_toxicity_dataset
- textmachinelab/quail
- tuetschek/atis
- xTRam1/safe-guard-prompt-injection
- yfxiao/folio-refined
- zeroshot/twitter-financial-news-topic
---
# jev pointer (Qwen3.5-2B, run 20261017-192041)

A "system one" model. It reads a **state** (a context), a **question** and a list of **options**, and in one
forward pass returns a probability for every option. It is meant for classification, routing, extraction checks,
NLI, guardrails and reading comprehension, where the answer follows from the text it is given. It is not meant to
answer from world knowledge.

- **Base model:** [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base). LoRA rank 16 on every
  attention, linear-attention and MLP projection, plus a 256-dim pointer head
  ([`src/jev_model/pointer/`](https://github.com/SergeyBazhmin/krasivo-jev/tree/v0.1.0/src/jev_model/pointer)).
- **Languages:** English and Russian.
- **Questions:** `noul` (yes/no) and `choice` (any number of options; there is no fixed slot count).
- **Training:** the `ce` stage only, 27,331 optimizer steps (2 epochs).
- **Run:** [`runs/pointer/20261017-192041`](https://github.com/SergeyBazhmin/krasivo-jev/tree/v0.1.0/runs/pointer/20261017-192041) in the training repo: config,
  training history and test scores.
- **Code:** [`krasivo-jev` @ `v0.1.0`](https://github.com/SergeyBazhmin/krasivo-jev/tree/v0.1.0), the release that includes this run.
- **Headline:** on the test splits of its 80 training datasets, macro accuracy is **0.879**, NLL 0.351 and ECE
  0.041. On 20 benchmark datasets it never trained on, macro accuracy is **0.640**, NLL 0.990 and ECE 0.123.

## How it works

The sample is laid out as one causal sequence. The delimiters reuse rarely used Qwen special tokens, so no
vocabulary rows are added:

```
<state> state <q> question <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
```

The pointer head projects the hidden state at `<decide>` and the hidden state at each `</opt>` to 256 dimensions.
An option's logit is their scaled dot product. A softmax over the options, at the fitted temperature (0.644 for
this run), gives the probabilities. Because the logit comes from each option's own text, any number of options
works and the option labels can be arbitrary. Sequences are cut to 4,096 tokens: a longer state loses its middle.

Attention is plain causal, so an option can see the options before it, and the output can depend somewhat on
option order. Training shuffles the options on every draw so that the model can't learn a position prior.

The design is adapted from [kev](https://github.com/jaredpalmer/kev). There is one question per sequence and no
option isolation, which suits the linear-attention (Gated DeltaNet) layers of Qwen3.5.

## Training

| setting | value |
|---|---|
| objective | focal loss (γ = 2) over the option distribution, which supports soft labels |
| optimizer steps | 27,331 (2 passes over the training set, `sized` sampling) |
| batch | up to 8,192 padded tokens × 2 accumulation steps |
| learning rate | 2e-4, 500 warmup steps, weight decay 0.01 |
| precision | bf16 base weights and autocast; LoRA and head in float32 |
| augmentation | options reshuffled on every draw. With p = 0.2, some wrong options are dropped (so the option count doesn't reveal the dataset). With p = 0.1, the true option's text is replaced by "None of the above" |
| model selection | every 2,500 steps, on up to 1,000 validation rows per dataset; the weights with the best macro NLL are kept, then one temperature is fitted on the whole validation partition |

With `sized` sampling, each pass shuffles the whole training set and packs it into token-budget batches. Every
sample is seen once per pass, so the datasets are mixed in proportion to their size.

To train it from scratch, you need at least 4 A100 GPUs; this run takes about 12–13 hours on 4 of them. Build the
data, then train with DDP from this run's config:

```bash
uv run jev prepare all --max-samples 25000 -j 8
uv run accelerate launch --multi_gpu --num_processes 4 -m jev_model.cli train pointer \
  -c runs/pointer/20261017-192041/config.json -s data_dir=data  # the saved data_dir is the training machine's path
```

![Validation curves](plots/training_curve.png)

## Data

All data is converted by the [`krasivo-jev`](https://github.com/SergeyBazhmin/krasivo-jev/tree/v0.1.0) pipeline ([`src/jev_datasets/`](https://github.com/SergeyBazhmin/krasivo-jev/tree/v0.1.0/src/jev_datasets)) into one format:
`state -> question -> options -> label`. The label is a probability distribution over the options: one-hot, or
annotator vote shares for sets such as ChaosNLI. Sources, subsets and rows that need outside knowledge were
dropped. Examples: SciQ keeps only rows that have a support passage, and BBH leaves out its knowledge-heavy tasks.
MMLU, ARC, OpenBookQA, CommonsenseQA, HellaSwag and WinoGrande are excluded for this reason. Each dataset uses
about 5 paraphrases of its question and of its instruction, picked deterministically per sample.

The model trained on the train splits of 80 datasets in 14 topics: bugs, business, classification, guardrails,
knowledge, logic, nlu, reasoning, retrieval, sentiment, spatial, structured, toxicity and truthfulness. The full
list, with sources, is in the tables below.

### Split method

Each dataset is saved with exactly `train`, `validation` and `test` splits, built as follows:

1. **Canonical names.** `dev`/`val` become `validation`; unknown split names become `train`.
2. **Disjoint contexts.** Rows are grouped by source document (an explicit `group` key when one document yields
   several passages or turns), or otherwise by a hash of their context (`state`, or `question` when there is no
   state). Each group moves whole into the first split it appears in, checked in the order test, validation,
   train. So **no context appears in two splits**, and a test passage is never trained on through another
   question about it. Exact duplicate rows are dropped, and the held-out copy is kept.
3. **Missing splits.** A missing validation or test split is carved from train by a hash of the group, 5% each.
   A source with no train split at all is re-split 80/10/10.
4. **Caps.** Each split is capped with a seeded random sample. This build used 25,000 rows per split.
5. **Sources that share data.** Many logic and spatial collections bundle each other (for example, SpaRP
   includes StepGame, and RuleCollection bundles ProofWriter and LogicNLI). Each underlying set is taken from
   one source only, so a benchmark's rows can't leak in through another dataset's train split.

Every step is deterministic, seeded from the sample content.

### Benchmarks

20 datasets are **benchmarks**. They are held out completely, and this model never saw their train split. They
are OpenDecisions (split into `noul` and `choice`), plus one converter per topic chosen by a seeded draw:
`ar_lsat`, `github_issue_type`, `hwu64`, `jailbreak_classification`, `logiqa`, the six `nomiracl_*` sets, `ropes`,
`rublimp`, `russian_toxicity`, `stepgame`, `tweet_sentiment`, `vitaminc` and `webnlg_en`. They measure transfer to
new tasks and formats, not fit to familiar ones.

## Evaluation

All scores are on the **test** split. Each row is scored by the option with the highest probability, at the
fitted temperature:

- **accuracy**: the argmax option is a correct one.
- **NLL**: the cross-entropy of the predicted distribution against the label distribution.
- **ECE**: expected calibration error of the top-option confidence, with 15 equal-width bins.
- **macro**: the mean over datasets, so every dataset counts the same however large it is.

The comparison model is TypeSafe's hosted `typesafe/jev-1.13`, called through its `system_one` API and scored on
the same test rows. The only exceptions are a few rows in `nomiracl_choice_*`, which were too long for this model's
4,096 tokens and were skipped. jev-1.13's training data is not known, so for that model neither the held-in sets nor
the benchmarks are guaranteed to be held out.

| | datasets | rows | accuracy | NLL | ECE | jev-1.13 accuracy | jev-1.13 NLL | jev-1.13 ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| held-in test | 80 | 294,408 | **0.879** | **0.351** | **0.041** | 0.829 | 0.882 | 0.080 |
| held-in, `noul` | 14 | | **0.933** | **0.165** | **0.027** | 0.858 | 0.326 | 0.052 |
| held-in, `choice` | 66 | | **0.868** | **0.391** | **0.044** | 0.823 | 1.000 | 0.085 |
| benchmarks | 20 | 102,170 | 0.640 | 0.990 | 0.123 | **0.825** | **0.748** | **0.067** |
| benchmarks, `noul` | 6 | | 0.724 | 0.515 | 0.095 | **0.896** | **0.254** | **0.051** |
| benchmarks, `choice` | 14 | | 0.604 | 1.193 | 0.135 | **0.795** | **0.960** | **0.074** |

All values are macro means over datasets; the better value in each pair is in bold.

![Accuracy by topic](plots/by_topic.png)

### Held-in datasets, against jev-1.13

On the test splits of its own training datasets, this model beats jev-1.13 on 55 of 80 datasets, and on average
its NLL is less than half of jev-1.13's. The biggest gains are on spatial QA (`spartqa_mchoice` +0.42, `spartqa_yn`
+0.34, `sparp` +0.29), emotion labels (`emotion` +0.34, `go_emotions` +0.26), `aegis_category` (+0.32) and
`halueval_summarization` (+0.29). It falls behind on multi-choice reading comprehension and argument reasoning:
`reclor` −0.33, `race` −0.13, `dream` −0.11 and `quail` −0.10. It is also behind on `tweet_hate` (−0.15) and on the
small Russian SuperGLUE sets `rcb` (−0.12) and `terra` (−0.07).

![Held-in macro accuracy by topic against jev-1.13](plots/by_topic_vs_jev.png)

![Held-in test accuracy](plots/test_accuracy.png)

### Benchmarks, against jev-1.13

![Benchmark accuracy](plots/benchmarks.png)

This model is strong on structured extraction checks, intent classification and guardrails, even on unseen
datasets (`webnlg_en` 0.995, `jailbreak_classification` 0.927, `hwu64` 0.895). It transfers poorly to multi-step
logic and spatial reasoning (`ar_lsat` 0.237, `stepgame` 0.350, `logiqa` 0.501), to Russian grammaticality
(`rublimp` 0.500, which is chance for its minimal pairs) and to OpenDecisions (about 0.5). Confidence is
over-stated on those sets, with ECE up to 0.32.

## Usage

The repo holds merged bf16 weights that vLLM can serve, plus the LoRA `adapter/` and `head.pt` for in-process
use. Both paths need the [`krasivo-jev`](https://github.com/SergeyBazhmin/krasivo-jev/tree/v0.1.0) package (`jev_model` and the `jev_vllm` plugins). Install it at the
[`v0.1.0`](https://github.com/SergeyBazhmin/krasivo-jev/releases/tag/v0.1.0) release, which matches these weights:

```bash
git clone https://github.com/SergeyBazhmin/krasivo-jev && cd krasivo-jev && git checkout v0.1.0
uv sync
```

```python
from jev_model.system_one import Choice, JevHolder, Noul

holder = JevHolder.from_pretrained("bizhmans/krasivo-jev-no-bench-bf16", revision="v0.1.0")
answers = holder.system_one(
    state="I was charged twice for my order. Please help.",
    questions={
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"angry": None, "calm": "polite"}),
    },
)
answers["billing"].noul  # P(yes)
answers["tone"].choice  # the likeliest key; also .confidence and .probabilities
```

```bash
uv run serve bizhmans/krasivo-jev-no-bench-bf16 -- --revision v0.1.0 --tokenizer-revision v0.1.0
# vLLM 0.17.1 + jev_vllm plugins in Docker, http://localhost:8080/pooling
curl localhost:8080/pooling -H 'Content-Type: application/json' -d '{"model": "pointer", "data": {
  "state": "I was charged twice. Please help.",
  "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}}'
```

Code written against `typesafe-sdk` can call the served model through
[`jev_vllm.client.JevClient`](https://github.com/SergeyBazhmin/krasivo-jev/blob/v0.1.0/src/jev_vllm/client.py), which takes the same question types and returns the SDK's
`SystemOneResponse`. Swap the client to switch between them:

```python
from typesafe_sdk import Choice, Noul
from jev_vllm.client import JevClient

with JevClient() as client:  # http://localhost:8080, model "pointer"
    response = client.system_one(
        state="I was charged twice for my order. Please help.",
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(instructions="What is the tone?", criteria={"angry": None, "calm": "polite"}),
        },
    )
response.nouls["billing"].noul, response.choices["tone"].choice
```

`AsyncJevClient` has the same `system_one`, awaited. vLLM batches concurrent requests itself, so `asyncio.gather` over
many calls keeps the GPU busy.

## Limitations

- The model is not a knowledge base. Questions whose answer is not in the state get confident-looking guesses.
- Option order can shift the probabilities slightly, because options attend causally to the ones before them.
- "None of the above" was trained only as an augmentation; the scores here don't measure it.
- Calibration holds on familiar tasks (macro ECE 0.041) but not on unfamiliar ones (0.123 on the benchmarks).
- The training mix is weighted by dataset size, so large English sets dominate it.

## Per-dataset scores

`n` is the number of test rows. `type` is `noul` (yes/no) or `choice`.

### Held-in (test)

| dataset | topic | type | source | n | accuracy | NLL | ECE | jev-1.13 acc | jev-1.13 NLL | jev-1.13 ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vscode_triage | bugs | choice | [Tilakoid/vscode-bug-feature-triage](https://huggingface.co/datasets/Tilakoid/vscode-bug-feature-triage) | 200 | 0.875 | 0.374 | 0.071 | 0.890 | 0.986 | 0.099 |
| financial_news_topic | business | choice | [zeroshot/twitter-financial-news-topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | 854 | 0.909 | 0.314 | 0.017 | 0.657 | 2.161 | 0.171 |
| ag_news | classification | choice | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | 7,600 | 0.927 | 0.214 | 0.027 | 0.889 | 0.894 | 0.067 |
| atis | classification | choice | [tuetschek/atis](https://huggingface.co/datasets/tuetschek/atis) | 848 | 0.986 | 0.055 | 0.011 | 0.929 | 0.256 | 0.022 |
| banking77 | classification | choice | [legacy-datasets/banking77](https://huggingface.co/datasets/legacy-datasets/banking77) | 3,080 | 0.931 | 0.274 | 0.013 | 0.799 | 1.705 | 0.087 |
| bitext_customer_support | classification | choice | [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | 1,217 | 1.000 | 0.002 | 0.002 | 0.921 | 0.285 | 0.024 |
| bitext_insurance | classification | choice | [bitext/Bitext-insurance-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset) | 1,934 | 0.999 | 0.002 | 0.001 | 0.896 | 0.391 | 0.020 |
| bitext_retail_banking | classification | choice | [bitext/Bitext-retail-banking-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset) | 1,246 | 0.998 | 0.006 | 0.002 | 0.954 | 0.374 | 0.030 |
| bitext_telco | classification | choice | [bitext/Bitext-telco-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset) | 1,284 | 0.998 | 0.004 | 0.002 | 0.921 | 0.292 | 0.038 |
| clinc_oos | classification | choice | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) `plus` | 5,502 | 0.925 | 0.292 | 0.028 | 0.885 | 0.726 | 0.024 |
| dbpedia | classification | choice | [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | 25,000 | 0.991 | 0.037 | 0.001 | 0.987 | 0.083 | 0.004 |
| massive_intent | classification | choice | [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) `en` | 2,972 | 0.898 | 0.437 | 0.019 | 0.744 | 1.761 | 0.111 |
| massive_intent_ru | classification | choice | [mteb/MassiveIntentClassification](https://huggingface.co/datasets/mteb/MassiveIntentClassification) `ru` | 2,925 | 0.878 | 0.490 | 0.026 | 0.777 | 1.739 | 0.108 |
| rus_med_dialogues | classification | choice | [Mykes/rus_med_dialogues](https://huggingface.co/datasets/Mykes/rus_med_dialogues) | 329 | 0.857 | 0.432 | 0.040 | 0.802 | 1.209 | 0.120 |
| rus_med_dialogues_noul | classification | noul | [Mykes/rus_med_dialogues](https://huggingface.co/datasets/Mykes/rus_med_dialogues) | 1,004 | 0.979 | 0.084 | 0.025 | 0.950 | 0.159 | 0.049 |
| ticket_routing | classification | choice | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | 4,093 | 0.442 | 1.572 | 0.069 | 0.278 | 7.707 | 0.441 |
| ticket_subject | classification | choice | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | 3,509 | 0.901 | 0.277 | 0.034 | 0.856 | 0.551 | 0.053 |
| ticket_type | classification | choice | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | 4,093 | 0.799 | 0.393 | 0.034 | 0.677 | 1.861 | 0.226 |
| aegis_category | guardrails | choice | [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | 672 | 0.717 | 0.880 | 0.050 | 0.396 | 3.819 | 0.346 |
| safeguard_prompt_injection | guardrails | noul | [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | 2,049 | 0.997 | 0.017 | 0.008 | 0.917 | 0.198 | 0.035 |
| spml_prompt_injection | guardrails | noul | [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | 852 | 0.996 | 0.016 | 0.003 | 0.897 | 0.286 | 0.025 |
| qasc | knowledge | choice | [allenai/qasc](https://huggingface.co/datasets/allenai/qasc) | 419 | 0.988 | 0.028 | 0.010 | 0.964 | 0.095 | 0.021 |
| quartz | knowledge | choice | [allenai/quartz](https://huggingface.co/datasets/allenai/quartz) | 784 | 0.904 | 0.263 | 0.025 | 0.952 | 0.615 | 0.037 |
| sciq | knowledge | choice | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | 886 | 0.997 | 0.011 | 0.007 | 0.999 | 0.007 | 0.002 |
| clutrr | logic | choice | [tasksource/clutrr](https://huggingface.co/datasets/tasksource/clutrr) | 1,045 | 0.448 | 4.125 | 0.410 | 0.414 | 2.174 | 0.187 |
| folio | logic | choice | [yfxiao/folio-refined](https://huggingface.co/datasets/yfxiao/folio-refined) | 39 | 0.744 | 0.581 | 0.141 | 0.769 | 1.183 | 0.195 |
| logicnli | logic | choice | [tasksource/LogicNLI](https://huggingface.co/datasets/tasksource/LogicNLI) | 1,999 | 0.730 | 0.625 | 0.044 | 0.606 | 2.330 | 0.205 |
| proofwriter | logic | choice | [tasksource/proofwriter](https://huggingface.co/datasets/tasksource/proofwriter) | 25,000 | 0.930 | 0.168 | 0.015 | 0.923 | 0.206 | 0.005 |
| proverqa | logic | choice | [opendatalab/ProverQA](https://huggingface.co/datasets/opendatalab/ProverQA) | 313 | 0.827 | 0.380 | 0.041 | 0.719 | 0.641 | 0.101 |
| anli | nlu | choice | [facebook/anli](https://huggingface.co/datasets/facebook/anli) | 8,204 | 0.852 | 0.395 | 0.014 | 0.865 | 0.453 | 0.039 |
| babi_nli | nlu | noul | [tasksource/babi_nli](https://huggingface.co/datasets/tasksource/babi_nli) | 11,934 | 0.970 | 0.087 | 0.023 | 0.924 | 0.210 | 0.011 |
| boolq | nlu | noul | [google/boolq](https://huggingface.co/datasets/google/boolq) | 392 | 0.847 | 0.366 | 0.055 | 0.926 | 0.215 | 0.035 |
| chaos_nli | nlu | choice | [tasksource/chaos-mnli-ambiguity](https://huggingface.co/datasets/tasksource/chaos-mnli-ambiguity) | 79 | 0.684 | 0.924 | 0.148 | 0.658 | 2.649 | 0.190 |
| danetqa | nlu | choice | [RussianNLP/russian_super_glue](https://huggingface.co/datasets/RussianNLP/russian_super_glue) `DaNetQA` | 67 | 0.910 | 0.271 | 0.076 | 0.896 | 0.207 | 0.066 |
| dream | nlu | choice | [dataset-org/dream](https://huggingface.co/datasets/dataset-org/dream) | 2,041 | 0.859 | 0.370 | 0.013 | 0.971 | 0.174 | 0.010 |
| hotpot_qa | nlu | choice | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) `distractor` | 325 | 0.951 | 0.157 | 0.044 | 0.951 | 0.181 | 0.027 |
| hotpot_qa_noul | nlu | noul | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) `distractor` | 264 | 0.902 | 0.247 | 0.050 | 0.943 | 0.184 | 0.034 |
| lidirus | nlu | choice | [RussianNLP/russian_super_glue](https://huggingface.co/datasets/RussianNLP/russian_super_glue) `LiDiRus` | 105 | 0.838 | 0.389 | 0.094 | 0.867 | 0.305 | 0.061 |
| mera_multiq | nlu | choice | [ai-forever/MERA](https://huggingface.co/datasets/ai-forever/MERA) `multiq` | 46 | 0.935 | 0.316 | 0.047 | 0.739 | 1.159 | 0.172 |
| multi_nli | nlu | choice | [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | 20,287 | 0.870 | 0.356 | 0.025 | 0.850 | 0.546 | 0.049 |
| multi_nli_genre | nlu | choice | [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | 6,623 | 0.931 | 0.192 | 0.006 | 0.841 | 0.697 | 0.060 |
| muserc | nlu | choice | [RussianNLP/russian_super_glue](https://huggingface.co/datasets/RussianNLP/russian_super_glue) `MuSeRC` | 536 | 0.882 | 0.285 | 0.034 | 0.937 | 0.464 | 0.042 |
| paws | nlu | noul | [google-research-datasets/paws](https://huggingface.co/datasets/google-research-datasets/paws) `labeled_final` | 7,987 | 0.942 | 0.164 | 0.024 | 0.860 | 0.322 | 0.022 |
| quail | nlu | choice | [textmachinelab/quail](https://huggingface.co/datasets/textmachinelab/quail) | 496 | 0.756 | 0.704 | 0.049 | 0.857 | 0.595 | 0.060 |
| race | nlu | choice | [ehovy/race](https://huggingface.co/datasets/ehovy/race) `all` | 4,934 | 0.801 | 0.555 | 0.036 | 0.929 | 0.369 | 0.022 |
| rcb | nlu | choice | [RussianNLP/russian_super_glue](https://huggingface.co/datasets/RussianNLP/russian_super_glue) `RCB` | 26 | 0.538 | 0.819 | 0.120 | 0.654 | 3.922 | 0.340 |
| ru_paraphraser | nlu | choice | [merionum/ru_paraphraser](https://huggingface.co/datasets/merionum/ru_paraphraser) | 1,906 | 0.724 | 0.637 | 0.030 | 0.716 | 0.714 | 0.102 |
| rucola | nlu | choice | [RussianNLP/rucola](https://huggingface.co/datasets/RussianNLP/rucola) | 357 | 0.773 | 0.527 | 0.113 | 0.796 | 0.835 | 0.126 |
| russe | nlu | choice | [RussianNLP/russian_super_glue](https://huggingface.co/datasets/RussianNLP/russian_super_glue) `RUSSE` | 550 | 0.662 | 0.625 | 0.032 | 0.715 | 0.648 | 0.095 |
| sberquad | nlu | choice | [kuznetsoffandrey/sberquad](https://huggingface.co/datasets/kuznetsoffandrey/sberquad) | 1,209 | 0.976 | 0.088 | 0.010 | 0.984 | 0.073 | 0.006 |
| terra | nlu | choice | [RussianNLP/russian_super_glue](https://huggingface.co/datasets/RussianNLP/russian_super_glue) `TERRa` | 106 | 0.849 | 0.372 | 0.078 | 0.915 | 0.200 | 0.045 |
| reclor | reasoning | choice | [tasksource/reclor](https://huggingface.co/datasets/tasksource/reclor) | 244 | 0.623 | 0.903 | 0.110 | 0.951 | 0.253 | 0.018 |
| wikiqa | retrieval | choice | [microsoft/wiki_qa](https://huggingface.co/datasets/microsoft/wiki_qa) | 615 | 0.737 | 0.707 | 0.062 | 0.727 | 1.706 | 0.148 |
| wikiqa_noul | retrieval | noul | [microsoft/wiki_qa](https://huggingface.co/datasets/microsoft/wiki_qa) | 1,587 | 0.900 | 0.252 | 0.023 | 0.895 | 0.250 | 0.016 |
| cedr | sentiment | choice | [sagteam/cedr_v1](https://huggingface.co/datasets/sagteam/cedr_v1) `main` | 9,378 | 0.952 | 0.132 | 0.027 | 0.918 | 0.316 | 0.030 |
| emotion | sentiment | choice | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) `split` | 2,014 | 0.933 | 0.125 | 0.008 | 0.591 | 3.516 | 0.258 |
| go_emotions | sentiment | choice | [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) `simplified` | 5,439 | 0.592 | 1.418 | 0.048 | 0.330 | 6.248 | 0.340 |
| go_emotions_noul | sentiment | noul | [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) `simplified` | 15,413 | 0.902 | 0.243 | 0.037 | 0.821 | 0.400 | 0.031 |
| imdb | sentiment | choice | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) | 24,801 | 0.962 | 0.137 | 0.051 | 0.964 | 0.194 | 0.018 |
| sst2 | sentiment | choice | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) | 3,338 | 0.951 | 0.166 | 0.051 | 0.931 | 0.222 | 0.016 |
| sparp | spatial | choice | [UKPLab/sparp](https://huggingface.co/datasets/UKPLab/sparp) `SpaRP-PS1 (SpaRTUN)` | 1,557 | 0.973 | 0.084 | 0.007 | 0.681 | 1.323 | 0.051 |
| spartqa_mchoice | spatial | choice | [tasksource/spartqa-mchoice](https://huggingface.co/datasets/tasksource/spartqa-mchoice) | 3,537 | 0.969 | 0.104 | 0.018 | 0.550 | 1.151 | 0.119 |
| spartqa_yn | spatial | choice | [tasksource/spartqa-yn](https://huggingface.co/datasets/tasksource/spartqa-yn) | 3,894 | 0.894 | 0.291 | 0.049 | 0.559 | 0.928 | 0.073 |
| atis_slots | structured | choice | [tuetschek/atis](https://huggingface.co/datasets/tuetschek/atis) | 1,816 | 0.998 | 0.011 | 0.004 | 0.997 | 0.012 | 0.008 |
| cuad_spans | structured | choice | [The-Atticus-Project/cuad](https://huggingface.co/datasets/The-Atticus-Project/cuad) | 259 | 1.000 | 0.001 | 0.001 | 1.000 | 0.000 | 0.000 |
| massive_slots | structured | choice | [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) `en-US` | 1,937 | 0.997 | 0.011 | 0.003 | 0.989 | 0.059 | 0.006 |
| massive_slots_ru | structured | choice | [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) `ru-RU` | 1,927 | 0.989 | 0.035 | 0.005 | 0.966 | 0.124 | 0.014 |
| multiwoz_state | structured | choice | [budzianowski/multiwoz](https://huggingface.co/datasets/budzianowski/multiwoz) `multiwoz` | 3,856 | 0.996 | 0.019 | 0.006 | 0.990 | 0.069 | 0.004 |
| nerel_relations | structured | choice | [nerel-ds/NEREL](https://huggingface.co/datasets/nerel-ds/NEREL) | 307 | 0.935 | 0.155 | 0.037 | 0.925 | 0.317 | 0.025 |
| sgd_state | structured | choice | [google-research-datasets/dstc8-schema-guided-dialogue](https://huggingface.co/datasets/google-research-datasets/dstc8-schema-guided-dialogue) `sgd` | 9,234 | 0.994 | 0.022 | 0.005 | 0.988 | 0.048 | 0.011 |
| skillspan | structured | choice | [kris927b/SkillSpan](https://huggingface.co/datasets/kris927b/SkillSpan) | 921 | 1.000 | 0.002 | 0.002 | 0.999 | 0.006 | 0.003 |
| totto_cells | structured | choice | [GEM/totto](https://huggingface.co/datasets/GEM/totto) | 4,882 | 0.995 | 0.015 | 0.001 | 0.999 | 0.002 | 0.001 |
| totto_statements | structured | choice | [GEM/totto](https://huggingface.co/datasets/GEM/totto) | 1,267 | 0.997 | 0.010 | 0.005 | 0.998 | 0.008 | 0.003 |
| beavertails | toxicity | noul | [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 14,163 | 0.748 | 0.507 | 0.056 | 0.714 | 0.685 | 0.143 |
| toxic_chat | toxicity | noul | [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) `toxicchat0124` | 4,976 | 0.968 | 0.089 | 0.026 | 0.974 | 0.080 | 0.017 |
| tweet_hate | toxicity | choice | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `hate` | 3,056 | 0.545 | 0.824 | 0.201 | 0.699 | 0.733 | 0.146 |
| tweet_offensive | toxicity | choice | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `offensive` | 850 | 0.847 | 0.381 | 0.085 | 0.815 | 0.529 | 0.082 |
| halueval_dialogue | truthfulness | noul | [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) `dialogue` | 994 | 0.909 | 0.209 | 0.041 | 0.665 | 0.633 | 0.166 |
| halueval_qa | truthfulness | noul | [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) `qa` | 954 | 0.996 | 0.025 | 0.010 | 0.824 | 0.360 | 0.053 |
| halueval_summarization | truthfulness | noul | [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) `summarization` | 970 | 0.999 | 0.009 | 0.003 | 0.708 | 0.579 | 0.093 |

### Benchmarks (test)

| dataset | topic | type | source | n | accuracy | NLL | ECE | jev-1.13 acc | jev-1.13 NLL | jev-1.13 ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| github_issue_type | bugs | choice | [khanmu2003/issue-classification](https://huggingface.co/datasets/khanmu2003/issue-classification) | 21,501 | 0.742 | 0.885 | 0.092 | 0.809 | 1.407 | 0.111 |
| open_decisions_choice | business | choice | [jgeuter/OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions) | 125 | 0.536 | 1.333 | 0.165 | 0.896 | 0.286 | 0.059 |
| open_decisions_noul | business | noul | [jgeuter/OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions) | 68 | 0.485 | 0.661 | 0.190 | 0.912 | 0.217 | 0.097 |
| hwu64 | classification | choice | [DeepPavlov/hwu64](https://huggingface.co/datasets/DeepPavlov/hwu64) | 1,076 | 0.895 | 0.326 | 0.032 | 0.818 | 1.157 | 0.055 |
| jailbreak_classification | guardrails | noul | [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | 262 | 0.927 | 0.187 | 0.027 | 0.962 | 0.125 | 0.026 |
| ropes | knowledge | choice | [allenai/ropes](https://huggingface.co/datasets/allenai/ropes) `plain_text` | 504 | 0.762 | 0.655 | 0.067 | 0.948 | 0.294 | 0.022 |
| ar_lsat | logic | choice | [RuleReasoner/RuleCollection-32K](https://huggingface.co/datasets/RuleReasoner/RuleCollection-32K) | 1,513 | 0.237 | 2.099 | 0.256 | 0.463 | 1.316 | 0.063 |
| rublimp | nlu | choice | [RussianNLP/RuBLiMP](https://huggingface.co/datasets/RussianNLP/RuBLiMP) | 4,570 | 0.500 | 0.738 | 0.094 | 0.969 | 0.097 | 0.033 |
| logiqa | reasoning | choice | [lucasmccabe/logiqa](https://huggingface.co/datasets/lucasmccabe/logiqa) | 651 | 0.501 | 1.383 | 0.141 | 0.779 | 1.116 | 0.076 |
| nomiracl_choice_en | retrieval | choice | [miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) `english` | 311 | 0.421 | 2.286 | 0.262 | 0.718 | 1.922 | 0.143 |
| nomiracl_choice_ru | retrieval | choice | [miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) `russian` | 227 | 0.476 | 1.921 | 0.188 | 0.736 | 1.183 | 0.099 |
| nomiracl_en | retrieval | noul | [miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) `english` | 313 | 0.818 | 0.500 | 0.164 | 0.875 | 0.313 | 0.053 |
| nomiracl_passage_en | retrieval | noul | [miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) `english` | 3,160 | 0.692 | 0.568 | 0.027 | 0.858 | 0.343 | 0.050 |
| nomiracl_passage_ru | retrieval | noul | [miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) `russian` | 2,358 | 0.645 | 0.623 | 0.021 | 0.891 | 0.260 | 0.020 |
| nomiracl_ru | retrieval | noul | [miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) `russian` | 231 | 0.775 | 0.550 | 0.139 | 0.879 | 0.267 | 0.059 |
| tweet_sentiment | sentiment | choice | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `sentiment` | 12,284 | 0.572 | 0.840 | 0.190 | 0.668 | 1.935 | 0.201 |
| stepgame | spatial | choice | [tasksource/stepgame](https://huggingface.co/datasets/tasksource/stepgame) | 25,000 | 0.350 | 3.006 | 0.316 | 0.548 | 1.785 | 0.082 |
| webnlg_en | structured | choice | [GEM/web_nlg](https://huggingface.co/datasets/GEM/web_nlg) `en` | 2,773 | 0.995 | 0.029 | 0.014 | 0.999 | 0.002 | 0.002 |
| russian_toxicity | toxicity | choice | [textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) | 243 | 0.749 | 0.473 | 0.058 | 0.963 | 0.084 | 0.015 |
| vitaminc | truthfulness | choice | [tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc) | 25,000 | 0.718 | 0.729 | 0.016 | 0.815 | 0.860 | 0.072 |
