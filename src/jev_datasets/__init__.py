from jev_datasets.bugs.github_issue_type import github_issue_type_dataset
from jev_datasets.bugs.vscode_triage import vscode_triage_dataset
from jev_datasets.business.financial_news_topic import financial_news_topic_dataset
from jev_datasets.business.open_decisions import open_decisions_choice_dataset, open_decisions_noul_dataset
from jev_datasets.classification.ag_news import ag_news_dataset
from jev_datasets.classification.atis import atis_dataset
from jev_datasets.classification.banking77 import banking77_dataset
from jev_datasets.classification.bitext import (
    bitext_customer_support_dataset,
    bitext_insurance_dataset,
    bitext_retail_banking_dataset,
    bitext_telco_dataset,
)
from jev_datasets.classification.clinc_oos import clinc_oos_dataset
from jev_datasets.classification.dbpedia import dbpedia_dataset
from jev_datasets.classification.hwu64 import hwu64_dataset
from jev_datasets.classification.massive_intent import massive_intent_dataset
from jev_datasets.classification.massive_intent_ru import massive_intent_ru_dataset
from jev_datasets.classification.rus_med_dialogues import rus_med_dialogues_dataset, rus_med_dialogues_noul_dataset
from jev_datasets.classification.ticket_routing import (
    ticket_routing_dataset,
    ticket_subject_dataset,
    ticket_type_dataset,
)
from jev_datasets.guardrails.aegis_category import aegis_category_dataset
from jev_datasets.guardrails.jailbreak_classification import jailbreak_classification_dataset
from jev_datasets.guardrails.safeguard_prompt_injection import safeguard_prompt_injection_dataset
from jev_datasets.guardrails.spml_prompt_injection import spml_prompt_injection_dataset
from jev_datasets.knowledge.qasc import qasc_dataset
from jev_datasets.knowledge.quartz import quartz_dataset
from jev_datasets.knowledge.ropes import ropes_dataset
from jev_datasets.knowledge.sciq import sciq_dataset
from jev_datasets.logic.clutrr import clutrr_dataset
from jev_datasets.logic.folio import folio_dataset
from jev_datasets.logic.logicnli import logicnli_dataset
from jev_datasets.logic.proofwriter import proofwriter_dataset
from jev_datasets.logic.proverqa import proverqa_dataset
from jev_datasets.logic.rule_collection import ar_lsat_dataset
from jev_datasets.nlu.anli import anli_dataset
from jev_datasets.nlu.babi_nli import babi_nli_dataset
from jev_datasets.nlu.boolq import boolq_dataset
from jev_datasets.nlu.chaos_nli import chaos_nli_dataset
from jev_datasets.nlu.dream import dream_dataset
from jev_datasets.nlu.hotpot_qa import hotpot_qa_dataset, hotpot_qa_noul_dataset
from jev_datasets.nlu.multi_nli import multi_nli_dataset, multi_nli_genre_dataset
from jev_datasets.nlu.paws import paws_dataset
from jev_datasets.nlu.quail import quail_dataset
from jev_datasets.nlu.race import race_dataset
from jev_datasets.nlu.ru_paraphraser import ru_paraphraser_dataset
from jev_datasets.nlu.rublimp import rublimp_dataset
from jev_datasets.nlu.rucola import rucola_dataset
from jev_datasets.nlu.russian_qa import mera_multiq_dataset, sberquad_dataset
from jev_datasets.nlu.russian_superglue import (
    danetqa_dataset,
    lidirus_dataset,
    muserc_dataset,
    rcb_dataset,
    russe_dataset,
    terra_dataset,
)
from jev_datasets.reasoning.logiqa import logiqa_dataset
from jev_datasets.reasoning.reclor import reclor_dataset
from jev_datasets.retrieval.nomiracl import (
    nomiracl_choice_en_dataset,
    nomiracl_choice_ru_dataset,
    nomiracl_en_dataset,
    nomiracl_passage_en_dataset,
    nomiracl_passage_ru_dataset,
    nomiracl_ru_dataset,
)
from jev_datasets.retrieval.wikiqa import wikiqa_dataset, wikiqa_noul_dataset
from jev_datasets.sentiment.cedr import cedr_dataset
from jev_datasets.sentiment.emotion import emotion_dataset
from jev_datasets.sentiment.go_emotions import go_emotions_dataset, go_emotions_noul_dataset
from jev_datasets.sentiment.imdb import imdb_dataset
from jev_datasets.sentiment.sst2 import sst2_dataset
from jev_datasets.sentiment.tweet_sentiment import tweet_sentiment_dataset
from jev_datasets.spatial.sparp import sparp_dataset
from jev_datasets.spatial.spartqa import spartqa_mchoice_dataset, spartqa_yn_dataset
from jev_datasets.spatial.stepgame import stepgame_dataset
from jev_datasets.structured.atis_slots import atis_slots_dataset
from jev_datasets.structured.cuad import cuad_spans_dataset
from jev_datasets.structured.dialogue_state import multiwoz_state_dataset, sgd_state_dataset
from jev_datasets.structured.massive_slots import massive_slots_dataset, massive_slots_ru_dataset
from jev_datasets.structured.nerel import nerel_relations_dataset
from jev_datasets.structured.skillspan import skillspan_dataset
from jev_datasets.structured.totto import totto_cells_dataset, totto_statements_dataset
from jev_datasets.structured.webnlg import webnlg_en_dataset
from jev_datasets.toxicity.beavertails import beavertails_dataset
from jev_datasets.toxicity.russian_toxicity import russian_toxicity_dataset
from jev_datasets.toxicity.toxic_chat import toxic_chat_dataset
from jev_datasets.toxicity.tweet_moderation import tweet_hate_dataset, tweet_offensive_dataset
from jev_datasets.truthfulness.halueval import (
    halueval_dialogue_dataset,
    halueval_qa_dataset,
    halueval_summarization_dataset,
)
from jev_datasets.truthfulness.vitaminc import vitaminc_dataset

datasets = {
    dataset.name: dataset
    for dataset in (
        aegis_category_dataset,
        ag_news_dataset,
        anli_dataset,
        ar_lsat_dataset,
        atis_dataset,
        atis_slots_dataset,
        babi_nli_dataset,
        banking77_dataset,
        beavertails_dataset,
        bitext_customer_support_dataset,
        bitext_insurance_dataset,
        bitext_retail_banking_dataset,
        bitext_telco_dataset,
        boolq_dataset,
        cedr_dataset,
        chaos_nli_dataset,
        clinc_oos_dataset,
        clutrr_dataset,
        cuad_spans_dataset,
        danetqa_dataset,
        dbpedia_dataset,
        dream_dataset,
        emotion_dataset,
        financial_news_topic_dataset,
        folio_dataset,
        github_issue_type_dataset,
        go_emotions_dataset,
        go_emotions_noul_dataset,
        halueval_dialogue_dataset,
        halueval_qa_dataset,
        halueval_summarization_dataset,
        hotpot_qa_dataset,
        hotpot_qa_noul_dataset,
        hwu64_dataset,
        imdb_dataset,
        jailbreak_classification_dataset,
        lidirus_dataset,
        logicnli_dataset,
        logiqa_dataset,
        massive_intent_dataset,
        massive_intent_ru_dataset,
        massive_slots_dataset,
        massive_slots_ru_dataset,
        mera_multiq_dataset,
        multi_nli_dataset,
        multi_nli_genre_dataset,
        multiwoz_state_dataset,
        muserc_dataset,
        nerel_relations_dataset,
        nomiracl_choice_en_dataset,
        nomiracl_choice_ru_dataset,
        nomiracl_en_dataset,
        nomiracl_passage_en_dataset,
        nomiracl_passage_ru_dataset,
        nomiracl_ru_dataset,
        open_decisions_choice_dataset,
        open_decisions_noul_dataset,
        paws_dataset,
        proofwriter_dataset,
        proverqa_dataset,
        qasc_dataset,
        quail_dataset,
        quartz_dataset,
        race_dataset,
        rcb_dataset,
        reclor_dataset,
        ropes_dataset,
        ru_paraphraser_dataset,
        rublimp_dataset,
        rucola_dataset,
        rus_med_dialogues_dataset,
        rus_med_dialogues_noul_dataset,
        russe_dataset,
        russian_toxicity_dataset,
        safeguard_prompt_injection_dataset,
        sberquad_dataset,
        sciq_dataset,
        sgd_state_dataset,
        skillspan_dataset,
        sparp_dataset,
        spartqa_mchoice_dataset,
        spartqa_yn_dataset,
        spml_prompt_injection_dataset,
        sst2_dataset,
        stepgame_dataset,
        terra_dataset,
        ticket_routing_dataset,
        ticket_subject_dataset,
        ticket_type_dataset,
        totto_cells_dataset,
        totto_statements_dataset,
        toxic_chat_dataset,
        tweet_hate_dataset,
        tweet_offensive_dataset,
        tweet_sentiment_dataset,
        vitaminc_dataset,
        vscode_triage_dataset,
        webnlg_en_dataset,
        wikiqa_dataset,
        wikiqa_noul_dataset,
    )
}
