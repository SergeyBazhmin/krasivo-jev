from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    disjoint_contexts,
    make_options,
    make_sample,
    pick_question,
    with_context,
)

QUESTIONS = [
    "Какое намерение выражает пользователь?",
    "Что пользователь хочет сделать с помощью ассистента?",
    "Какому намерению соответствует эта команда?",
    "Какова цель данного запроса к ассистенту?",
    "Как следует классифицировать намерение пользователя?",
]
INTENTS = {
    "alarm_query": "узнать об установленных будильниках",
    "alarm_remove": "удалить будильник",
    "alarm_set": "установить будильник",
    "audio_volume_down": "уменьшить громкость",
    "audio_volume_mute": "выключить звук",
    "audio_volume_other": "другое управление громкостью",
    "audio_volume_up": "увеличить громкость",
    "calendar_query": "узнать о событиях календаря",
    "calendar_remove": "удалить событие календаря",
    "calendar_set": "создать событие календаря",
    "cooking_query": "получить информацию о приготовлении еды",
    "cooking_recipe": "получить рецепт блюда",
    "datetime_convert": "перевести время между часовыми поясами",
    "datetime_query": "узнать дату или время",
    "email_addcontact": "добавить контакт электронной почты",
    "email_query": "проверить электронную почту",
    "email_querycontact": "узнать данные почтового контакта",
    "email_sendemail": "отправить электронное письмо",
    "general_greet": "поприветствовать ассистента",
    "general_joke": "попросить рассказать шутку",
    "general_quirky": "поговорить с ассистентом на отвлечённую тему",
    "iot_cleaning": "управлять устройством для уборки",
    "iot_coffee": "управлять кофеваркой",
    "iot_hue_lightchange": "изменить цвет освещения",
    "iot_hue_lightdim": "приглушить освещение",
    "iot_hue_lightoff": "выключить освещение",
    "iot_hue_lighton": "включить освещение",
    "iot_hue_lightup": "увеличить яркость освещения",
    "iot_wemo_off": "выключить устройство через умную розетку",
    "iot_wemo_on": "включить устройство через умную розетку",
    "lists_createoradd": "создать список или добавить в него пункт",
    "lists_query": "просмотреть список",
    "lists_remove": "удалить список или пункт списка",
    "music_dislikeness": "выразить неприязнь к музыке",
    "music_likeness": "выразить симпатию к музыке",
    "music_query": "узнать информацию о музыке",
    "music_settings": "изменить настройки воспроизведения музыки",
    "news_query": "узнать новости",
    "play_audiobook": "включить аудиокнигу",
    "play_game": "начать игру",
    "play_music": "включить музыку",
    "play_podcasts": "включить подкаст",
    "play_radio": "включить радио",
    "qa_currency": "узнать курс валют или перевести сумму в другую валюту",
    "qa_definition": "узнать определение слова или понятия",
    "qa_factoid": "задать вопрос о факте",
    "qa_maths": "попросить выполнить математическое вычисление",
    "qa_stock": "узнать биржевую информацию",
    "recommendation_events": "получить рекомендацию мероприятий",
    "recommendation_locations": "получить рекомендацию мест",
    "recommendation_movies": "получить рекомендацию фильмов",
    "social_post": "опубликовать сообщение в социальной сети",
    "social_query": "просмотреть сообщения в социальной сети",
    "takeaway_order": "заказать еду с доставкой или навынос",
    "takeaway_query": "узнать информацию о заказе еды",
    "transport_query": "узнать информацию о транспорте",
    "transport_taxi": "заказать такси",
    "transport_ticket": "купить транспортный билет",
    "transport_traffic": "узнать дорожную обстановку",
    "weather_query": "узнать погоду",
}
ABOUT = (
    "Определите намерение пользователя по его команде виртуальному ассистенту. "
    "Варианты описывают действия ассистента. Классифицируйте запрос, не выполняя его и не отвечая на вопросы о фактах."
)


class MassiveIntentRuDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        ids = sorted(INTENTS)
        options = make_options(ids, [INTENTS[id] for id in ids])
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Команда"), pick_question(QUESTIONS, x["text"]), options, x["label"]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )
        self.data = disjoint_contexts(self.data)


massive_intent_ru_dataset = MassiveIntentRuDataset(
    name="massive_intent_ru", hf_path="mteb/MassiveIntentClassification", hf_name="ru"
)
