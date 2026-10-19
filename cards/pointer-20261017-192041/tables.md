## Held-in
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

## Benchmarks
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

- held-in noul: 14 datasets, macro acc 0.933, NLL 0.165, ECE 0.027
- held-in choice: 66 datasets, macro acc 0.868, NLL 0.391, ECE 0.044
- benchmark noul: 6 datasets, macro acc 0.724, NLL 0.515, ECE 0.095
- benchmark choice: 14 datasets, macro acc 0.604, NLL 1.193, ECE 0.135
- jev-1.13 held-in: 80 datasets, macro acc 0.829, NLL 0.882, ECE 0.080
- jev-1.13 benchmark: 20 datasets, macro acc 0.825, NLL 0.748, ECE 0.067
