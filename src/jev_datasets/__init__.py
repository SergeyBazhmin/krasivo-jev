from jev_datasets.classification.ag_news import ag_news_dataset
from jev_datasets.classification.banking77 import banking77_dataset
from jev_datasets.classification.ru_massive_intent import ru_massive_intent_dataset
from jev_datasets.classification.rus_med_dialogues import rus_med_dialogues_dataset
from jev_datasets.classification.ticket_routing import ticket_routing_dataset
from jev_datasets.commonsense.commonsense_qa import commonsense_qa_dataset
from jev_datasets.commonsense.hellaswag import hellaswag_dataset
from jev_datasets.commonsense.ru_parus import ru_parus_dataset
from jev_datasets.commonsense.ru_rwsd import ru_rwsd_dataset
from jev_datasets.commonsense.winogrande import winogrande_dataset
from jev_datasets.distill.jev_distill import jev_distill_dataset
from jev_datasets.judge.helpsteer2 import helpsteer2_dataset
from jev_datasets.judge.hh_rlhf import hh_rlhf_dataset
from jev_datasets.judge.pku_saferlhf import pku_saferlhf_dataset
from jev_datasets.judge.ultrafeedback import ultrafeedback_dataset
from jev_datasets.knowledge.arc import arc_challenge_dataset, arc_easy_dataset
from jev_datasets.knowledge.mera import ru_mmlu_dataset, ru_openbookqa_dataset, ru_worldtree_dataset
from jev_datasets.knowledge.mmlu import mmlu_dataset
from jev_datasets.knowledge.openbookqa import openbookqa_dataset
from jev_datasets.knowledge.sciq import sciq_dataset
from jev_datasets.math.aqua_rat import aqua_rat_dataset
from jev_datasets.math.mmlu_pro_math import mmlu_pro_math_dataset
from jev_datasets.math.ru_mathlogicqa import ru_mathlogicqa_dataset
from jev_datasets.nlu.anli import anli_dataset
from jev_datasets.nlu.boolq import boolq_dataset
from jev_datasets.nlu.chaos_nli import chaos_nli_dataset
from jev_datasets.nlu.multi_nli import multi_nli_dataset
from jev_datasets.nlu.race import race_dataset
from jev_datasets.nlu.ru_danetqa import ru_danetqa_dataset
from jev_datasets.nlu.ru_rcb import ru_rcb_dataset
from jev_datasets.nlu.ru_terra import ru_terra_dataset
from jev_datasets.reasoning.bbh import bbh_dataset
from jev_datasets.reasoning.logiqa import logiqa_dataset
from jev_datasets.reasoning.reclor import reclor_dataset
from jev_datasets.sentiment.emotion import emotion_dataset
from jev_datasets.sentiment.go_emotions import go_emotions_dataset
from jev_datasets.sentiment.imdb import imdb_dataset
from jev_datasets.sentiment.sst2 import sst2_dataset
from jev_datasets.sentiment.tweet_sentiment import tweet_sentiment_dataset
from jev_datasets.toxicity.beavertails import beavertails_dataset
from jev_datasets.toxicity.civil_comments import civil_comments_dataset
from jev_datasets.toxicity.multilingual_toxicity import multilingual_toxicity_dataset
from jev_datasets.toxicity.toxic_chat import toxic_chat_dataset
from jev_datasets.toxicity.tweet_moderation import tweet_hate_dataset, tweet_offensive_dataset
from jev_datasets.truthfulness.halueval import halueval_dialogue_dataset, halueval_general_dataset, halueval_qa_dataset, halueval_summarization_dataset
from jev_datasets.truthfulness.truthful_qa import truthful_qa_mc1_dataset, truthful_qa_mc2_dataset

datasets = {
    dataset.name: dataset
    for dataset in (
        ag_news_dataset,
        anli_dataset,
        aqua_rat_dataset,
        arc_challenge_dataset,
        arc_easy_dataset,
        banking77_dataset,
        bbh_dataset,
        beavertails_dataset,
        boolq_dataset,
        chaos_nli_dataset,
        civil_comments_dataset,
        commonsense_qa_dataset,
        emotion_dataset,
        go_emotions_dataset,
        halueval_dialogue_dataset,
        halueval_general_dataset,
        halueval_qa_dataset,
        halueval_summarization_dataset,
        hellaswag_dataset,
        helpsteer2_dataset,
        hh_rlhf_dataset,
        imdb_dataset,
        jev_distill_dataset,
        logiqa_dataset,
        mmlu_dataset,
        mmlu_pro_math_dataset,
        multi_nli_dataset,
        multilingual_toxicity_dataset,
        openbookqa_dataset,
        pku_saferlhf_dataset,
        race_dataset,
        reclor_dataset,
        ru_danetqa_dataset,
        ru_massive_intent_dataset,
        ru_mathlogicqa_dataset,
        ru_mmlu_dataset,
        ru_openbookqa_dataset,
        ru_parus_dataset,
        ru_rcb_dataset,
        ru_rwsd_dataset,
        ru_terra_dataset,
        ru_worldtree_dataset,
        rus_med_dialogues_dataset,
        sciq_dataset,
        sst2_dataset,
        ticket_routing_dataset,
        toxic_chat_dataset,
        truthful_qa_mc1_dataset,
        truthful_qa_mc2_dataset,
        tweet_hate_dataset,
        tweet_offensive_dataset,
        tweet_sentiment_dataset,
        ultrafeedback_dataset,
        winogrande_dataset,
    )
}
