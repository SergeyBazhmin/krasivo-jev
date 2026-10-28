from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Является ли это SMS спамом?",
    "Это спам?",
    "Следует ли пометить сообщение как спам?",
    "Это рекламная рассылка, а не личное сообщение?",
    "Нужно ли отправить это SMS в папку «Спам»?",
]
OPTIONS = make_options(["yes", "no"], ["да", "нет"])
ABOUT = [
    (
        "Вы фильтруете входящие SMS. Спам — массовые непрошеные сообщения: розыгрыши призов, платные подписки, "
        "предложения перезвонить или отправить SMS на короткий номер, реклама услуг. Личная переписка, даже "
        "бессвязная или с ошибками, спамом не является."
    ),
    (
        "Ниже SMS-сообщение. Решите, спам ли это: непрошеная рассылка с выигрышами и призами, платными подписками, "
        "призывами позвонить или ответить на короткий номер, рекламой. Обычные сообщения между знакомыми, даже "
        "сумбурные или с опечатками, спамом не считаются."
    ),
    (
        "Определите, является ли SMS спамом. Спам — это массовая реклама и приманки: «вы выиграли приз», платные "
        "услуги и подписки, просьбы позвонить или написать на короткий номер. Личные сообщения, даже неразборчивые, "
        "спамом не являются."
    ),
    (
        "Вы антиспам-фильтр для SMS. Спамом считаются непрошеные рассылки: призы и розыгрыши, платные подписки, "
        "реклама, предложения позвонить или отправить сообщение на короткий номер. Переписка знакомых людей, даже "
        "с ошибками или сленгом, — не спам."
    ),
    (
        "Оцените SMS-сообщение. Спам — массовые рекламные сообщения: выигрыши, призы, платные подписки, призывы "
        "перезвонить или ответить на короткий номер. Обычные личные SMS, пусть сумбурные или с опечатками, спамом не "
        "являются."
    ),
]


class AntiSpamRuDataset(JevDataset):
    type = DatasetType.NOUL
    # its topic's benchmark pick: scored by every model, trained on only with `train_on_benchmark`
    benchmark = True

    def prepare(self):
        # A machine translation of the UCI SMS Spam Collection (hence the odd phrasing). Don't also add
        # `ucirvine/sms_spam`: the same messages would sit in both datasets' splits.
        self.data = self.data.filter(lambda x: bool(x["text"].strip()))
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "SMS"),
                pick_question(QUESTIONS, x["text"]),
                OPTIONS,
                "yes" if x["is_spam"] else "no",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


anti_spam_ru_dataset = AntiSpamRuDataset(name="anti_spam_ru", hf_path="DmitryKRX/anti_spam_ru")
