from jev_datasets.bugs.github_issue_type import github_issue_type_dataset
from jev_datasets.bugs.vscode_triage import vscode_triage_dataset
from jev_datasets.business.financial_news_topic import financial_news_topic_dataset
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
from jev_datasets.classification.ticket_routing import ticket_routing_dataset, ticket_type_dataset
from jev_datasets.guardrails.aegis_safety import aegis_safety_dataset
from jev_datasets.guardrails.deepset_prompt_injections import deepset_prompt_injections_dataset
from jev_datasets.guardrails.jailbreak_classification import jailbreak_classification_dataset
from jev_datasets.guardrails.safeguard_prompt_injection import safeguard_prompt_injection_dataset
from jev_datasets.guardrails.spml_prompt_injection import spml_prompt_injection_dataset
from jev_datasets.knowledge.sciq import sciq_dataset
from jev_datasets.logic.clutrr import clutrr_dataset
from jev_datasets.logic.folio import folio_dataset
from jev_datasets.logic.logicnli import logicnli_dataset
from jev_datasets.logic.proofwriter import proofwriter_dataset
from jev_datasets.logic.proverqa import proverqa_dataset
from jev_datasets.logic.rule_collection import ar_lsat_dataset, prontoqa_dataset
from jev_datasets.nlu.anli import anli_dataset
from jev_datasets.nlu.babi_nli import babi_nli_dataset
from jev_datasets.nlu.boolq import boolq_dataset
from jev_datasets.nlu.chaos_nli import chaos_nli_dataset
from jev_datasets.nlu.multi_nli import multi_nli_dataset, multi_nli_genre_dataset
from jev_datasets.nlu.paws import paws_dataset
from jev_datasets.nlu.race import race_dataset
from jev_datasets.reasoning.bbh import bbh_dataset
from jev_datasets.reasoning.logiqa import logiqa_dataset
from jev_datasets.reasoning.reclor import reclor_dataset
from jev_datasets.sentiment.emotion import emotion_dataset
from jev_datasets.sentiment.go_emotions import go_emotions_dataset
from jev_datasets.sentiment.imdb import imdb_dataset
from jev_datasets.sentiment.sst2 import sst2_dataset
from jev_datasets.sentiment.tweet_sentiment import tweet_sentiment_dataset
from jev_datasets.spatial.sparp import sparp_dataset
from jev_datasets.spatial.spartqa import spartqa_mchoice_dataset, spartqa_yn_dataset
from jev_datasets.spatial.stepgame import stepgame_dataset
from jev_datasets.toxicity.beavertails import beavertails_dataset
from jev_datasets.toxicity.civil_comments import civil_comments_dataset
from jev_datasets.toxicity.toxic_chat import toxic_chat_dataset
from jev_datasets.toxicity.tweet_moderation import tweet_hate_dataset, tweet_offensive_dataset
from jev_datasets.truthfulness.halueval import halueval_dialogue_dataset, halueval_qa_dataset, halueval_summarization_dataset

datasets = {
    dataset.name: dataset
    for dataset in (
        aegis_safety_dataset,
        ag_news_dataset,
        anli_dataset,
        ar_lsat_dataset,
        atis_dataset,
        babi_nli_dataset,
        banking77_dataset,
        bbh_dataset,
        beavertails_dataset,
        bitext_customer_support_dataset,
        bitext_insurance_dataset,
        bitext_retail_banking_dataset,
        bitext_telco_dataset,
        boolq_dataset,
        chaos_nli_dataset,
        civil_comments_dataset,
        clinc_oos_dataset,
        clutrr_dataset,
        dbpedia_dataset,
        deepset_prompt_injections_dataset,
        emotion_dataset,
        financial_news_topic_dataset,
        folio_dataset,
        github_issue_type_dataset,
        go_emotions_dataset,
        halueval_dialogue_dataset,
        halueval_qa_dataset,
        halueval_summarization_dataset,
        hwu64_dataset,
        imdb_dataset,
        jailbreak_classification_dataset,
        logicnli_dataset,
        logiqa_dataset,
        massive_intent_dataset,
        multi_nli_dataset,
        multi_nli_genre_dataset,
        paws_dataset,
        prontoqa_dataset,
        proofwriter_dataset,
        proverqa_dataset,
        race_dataset,
        reclor_dataset,
        safeguard_prompt_injection_dataset,
        sciq_dataset,
        sparp_dataset,
        spartqa_mchoice_dataset,
        spartqa_yn_dataset,
        spml_prompt_injection_dataset,
        sst2_dataset,
        stepgame_dataset,
        ticket_routing_dataset,
        ticket_type_dataset,
        toxic_chat_dataset,
        tweet_hate_dataset,
        tweet_offensive_dataset,
        tweet_sentiment_dataset,
        vscode_triage_dataset,
    )
}
