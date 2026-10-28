from collections import defaultdict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    explode,
    make_options,
    make_sample,
    pick_question,
    stable_shuffle,
    with_context,
)

# Synthetic texts with every personal-data span marked. Each sample names one span and asks for its type.
QUESTIONS = {
    "en": [
        "What kind of personal data is «{value}» in this text?",
        "Which type of personal information does «{value}» represent?",
        "Classify the personal data «{value}» found in the text.",
        "What is «{value}» in this text?",
        "Which category of personal data does «{value}» belong to?",
    ],
    "ru": [
        "К какому типу персональных данных относится «{value}» в этом тексте?",
        "Что за персональные данные «{value}»?",
        "Определите тип персональных данных «{value}» в тексте.",
        "Чем является «{value}» в этом тексте?",
        "К какой категории персональных данных относится «{value}»?",
    ],
}
ABOUT = {
    "en": [
        (
            "You find personal data in texts before they are shared. Decide what type of personal data a fragment is, "
            "judging by its form and by how the text uses it."
        ),
        (
            "Below is a text containing personal data. Determine the type of the given fragment from its format and "
            "from the way the text refers to it."
        ),
        (
            "You are a PII detector. For the given fragment of the text, choose its type of personal information, "
            "based on its format and its role in the text."
        ),
        (
            "A text with personal information is shown below. Classify the given fragment by type of personal data, "
            "relying on its form and on the surrounding words."
        ),
        (
            "Label the personal data in the text below: say what type the given fragment is, as shown by its format and "
            "by how the text uses it."
        ),
    ],
    "ru": [
        (
            "Вы находите персональные данные в текстах перед их публикацией. Определите тип указанного фрагмента по его "
            "формату и по тому, как он употреблён в тексте."
        ),
        (
            "Ниже текст с персональными данными. Определите, к какому типу относится указанный фрагмент, по его формату "
            "и по контексту."
        ),
        (
            "Вы детектор персональных данных. Для указанного фрагмента текста выберите тип персональных данных, исходя "
            "из его формата и роли в тексте."
        ),
        (
            "Перед вами текст, содержащий персональные данные. Классифицируйте указанный фрагмент по типу, опираясь на "
            "его вид и окружающие слова."
        ),
        (
            "Разметьте персональные данные в тексте ниже: укажите тип заданного фрагмента, судя по его формату и по "
            "тому, как он используется."
        ),
    ],
}
# SEX and GENDER are one option: the source switches between them for the same kind of value («Masculin»).
# Each type is (en name, ru name, en meaning, ru meaning). The meanings separate the look-alike types (a date from a
# date of birth, an ID card from a passport or a licence), so the answer follows from the text, not from the tag set.
TYPES = {
    "GIVENNAME": (
        "first name",
        "имя",
        "a person's own name, such as John or Maria, not the family name",
        "личное имя человека, например Иван или Мария, но не фамилия",
    ),
    "SURNAME": (
        "surname",
        "фамилия",
        "a person's family name, such as Smith",
        "фамилия человека, например Петров",
    ),
    "TITLE": (
        "title or form of address",
        "обращение или титул",
        "a word put before a name: Mr, Mrs, Ms, Dr, Prof, Sir and the like",
        "слово перед именем: господин, госпожа, доктор, профессор и т. п.",
    ),
    "SEX": (
        "sex or gender",
        "пол или гендер",
        "a word stating a person's sex or gender: male, female, man, woman, non-binary",
        "слово, указывающее пол человека: мужской, женский, мужчина, женщина",
    ),
    "AGE": (
        "age",
        "возраст",
        "how old a person is, usually a number of years",
        "сколько человеку лет, обычно число лет",
    ),
    "DATEOFBIRTH": (
        "date of birth",
        "дата рождения",
        "a date the text gives as the day a person was born",
        "дата, которую текст называет днём рождения человека",
    ),
    "DATE": (
        "date",
        "дата",
        "any other calendar date: of a meeting, a payment, a visit",
        "любая другая календарная дата: встречи, платежа, визита",
    ),
    "TIME": (
        "time",
        "время",
        "a time of day, such as 14:30 or 9 am",
        "время суток, например 14:30 или 9 утра",
    ),
    "CITY": (
        "city or town",
        "населённый пункт",
        "the name of a city, town or village",
        "название города, посёлка или села",
    ),
    "STREET": (
        "street",
        "улица",
        "the name of a street, avenue, road or square, without the building number",
        "название улицы, проспекта, переулка или площади без номера дома",
    ),
    "BUILDINGNUM": (
        "building number",
        "номер дома",
        "the number of a house or building in an address",
        "номер дома или строения в адресе",
    ),
    "ZIPCODE": (
        "postal code",
        "почтовый индекс",
        "the postal or ZIP code of an address",
        "почтовый индекс адреса, обычно шесть цифр",
    ),
    "TELEPHONENUM": (
        "phone number",
        "номер телефона",
        "a telephone number, often with a country code, brackets or dashes",
        "номер телефона, часто с кодом страны (+7, 8), скобками или дефисами",
    ),
    "EMAIL": (
        "email address",
        "адрес электронной почты",
        "an address of the form name@domain",
        "адрес вида имя@домен",
    ),
    "USERNAME": (
        "username",
        "имя пользователя",
        "a login, nickname or handle on a site or service, not an email address",
        "логин, ник или учётное имя на сайте или в сервисе, но не адрес почты",
    ),
    "PASSWORD": (
        "password",
        "пароль",
        "a secret string used to log in, often a mix of letters, digits and symbols",
        "секретная строка для входа, часто смесь букв, цифр и символов",
    ),
    "ACCOUNTNUM": (
        "bank account number",
        "номер банковского счёта",
        "the number of a bank account or IBAN, not of a card",
        "номер банковского счёта или IBAN, но не карты",
    ),
    "CREDITCARDNUMBER": (
        "credit card number",
        "номер банковской карты",
        "the number printed on a payment card, usually 16 digits in groups of four",
        "номер платёжной карты, обычно 16 цифр группами по четыре",
    ),
    "IDCARDNUM": (
        "ID card number",
        "номер удостоверения личности",
        "the number of a national ID card or other identity document that is not a passport or a driver's license",
        "номер удостоверения личности или другого документа, кроме паспорта и водительских прав",
    ),
    "PASSPORTNUM": (
        "passport number",
        "номер паспорта",
        "the number (or series and number) of a passport",
        "номер паспорта, иногда вместе с серией",
    ),
    "DRIVERLICENSENUM": (
        "driver's license number",
        "номер водительского удостоверения",
        "the number of a driver's license",
        "номер водительского удостоверения (прав)",
    ),
    "SOCIALNUM": (
        "social security number",
        "номер социального страхования",
        "a social security or national insurance number, such as a US SSN",
        "номер социального страхования, например СНИЛС",
    ),
    "TAXNUM": (
        "tax ID number",
        "ИНН",
        "a taxpayer identification number",
        "идентификационный номер налогоплательщика",
    ),
}
ALIASES = {"GENDER": "SEX"}
MAX_PER_TEXT = 3


class PiiMaskingDataset(JevDataset):
    type = DatasetType.CHOICE

    def __init__(self, name: str, hf_path: str, language: str):
        super().__init__(name=name, hf_path=hf_path)
        self.language = language

    def prepare(self):
        index = 0 if self.language == "en" else 1
        options = make_options(list(TYPES), [texts[index] for texts in TYPES.values()])
        answers = {texts[index]: texts[index + 2] for texts in TYPES.values()}
        names = ("Text", "Answers") if self.language == "en" else ("Текст", "Ответы")

        def convert(row: dict) -> list[dict]:
            text = row["source_text"]
            kinds = defaultdict(set)
            for span in row["privacy_mask"]:
                kinds[span["value"].strip()].add(ALIASES.get(span["label"], span["label"]))
            # A value marked with two types (or an unmapped one) has no single answer.
            spans = sorted((value, *labels) for value, labels in kinds.items() if len(labels) == 1 and value)
            spans = [(value, label) for value, label in spans if label in TYPES]
            state = with_context(ABOUT[self.language], text, names[0], answers, names[1])
            return [
                make_sample(state, pick_question(QUESTIONS[self.language], value).format(value=value), options, label)
                for value, label in stable_shuffle(spans, text)[:MAX_PER_TEXT]
            ]

        if self.language == "en":
            # The 500k set mixes eight languages; the Russian side comes from `ru_pii` instead.
            self.data = self.data.filter(lambda x: x["language"] == "en")
        self.data = self.data.map(
            explode(convert), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


pii_masking_en_dataset = PiiMaskingDataset(
    name="pii_masking_en", hf_path="ai4privacy/open-pii-masking-500k-ai4privacy", language="en"
)
ru_pii_dataset = PiiMaskingDataset(name="ru_pii", hf_path="wolframko/russian-pii-66k", language="ru")
