import json
import re
import tarfile
from collections import defaultdict

from datasets import Dataset, DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, disjoint_contexts, explode, pick_question, stable_shuffle, text_choice

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
QUESTIONS = [
    "Which phrase in the command specifies {slot}?",
    "What does the user say for {slot}?",
    "Choose the value of {slot} in this request.",
    "Which value does the command give for {slot}?",
    "Identify {slot} as stated by the user.",
]
QUESTIONS_RU = [
    "Какой фрагмент команды задаёт {slot}?",
    "Что пользователь указал для поля «{slot}»?",
    "Выберите значение поля «{slot}» из этой команды.",
    "Какое значение поля «{slot}» содержится в запросе?",
    "Укажите {slot}, используя слова пользователя.",
]
# Concrete spans only: vague descriptors and personal_info need an unexplained source ontology.
SLOTS = {
    "time": ("the time or time interval", "время или интервал времени"),
    "date": ("the date or day", "дату или день"),
    "person": ("the person's name", "имя человека"),
    "place_name": ("the place name", "название места"),
    "artist_name": ("the artist's name", "имя исполнителя"),
    "song_name": ("the song title", "название песни"),
    "music_album": ("the album title", "название альбома"),
    "playlist_name": ("the playlist name", "название плейлиста"),
    "podcast_name": ("the podcast title", "название подкаста"),
    "radio_name": ("the radio station name", "название радиостанции"),
    "audiobook_name": ("the audiobook title", "название аудиокниги"),
    "audiobook_author": ("the audiobook author", "автора аудиокниги"),
    "movie_name": ("the movie title", "название фильма"),
    "app_name": ("the application name", "название приложения"),
    "business_name": ("the business name", "название организации"),
    "email_address": ("the email address", "адрес электронной почты"),
    "email_folder": ("the email folder", "папку электронной почты"),
    "event_name": ("the event name", "название события"),
    "list_name": ("the list name", "название списка"),
    "device_type": ("the device", "устройство"),
    "house_place": ("the room or place in the home", "комнату или место в доме"),
    "color_type": ("the color", "цвет"),
    "ingredient": ("the ingredient", "ингредиент"),
    "currency_name": ("the currency", "валюту"),
    "time_zone": ("the time zone", "часовой пояс"),
    "transport_agency": ("the transport operator", "транспортного оператора"),
    "transport_type": ("the means of transport", "вид транспорта"),
}
ANNOTATION = re.compile(r"\[([^:\[\]]+)\s+:\s+([^\[\]]+)\]")


def spans(row: dict) -> list[tuple[str, str]]:
    annotated = row["annot_utt"]
    # Reject broken annotations rather than treating the annotation markup as user input.
    if ANNOTATION.sub(lambda match: match[2], annotated) != row["utt"]:
        return []
    return [(match[1].strip(), match[2]) for match in ANNOTATION.finditer(annotated)]


class MassiveSlotsDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        path = DownloadManager().download(URL)
        with tarfile.open(path) as archive:
            member = next(member for member in archive.getmembers() if member.name.endswith(f"{self.hf_name}.jsonl"))
            rows = [json.loads(line) for line in archive.extractfile(member)]
        return DatasetDict(
            {
                split: Dataset.from_list([row for row in rows if row["partition"] == source_split])
                for split, source_split in (("train", "train"), ("validation", "dev"), ("test", "test"))
            }
        )

    def prepare(self):
        russian = self.hf_name == "ru-RU"
        pools = defaultdict(set)
        # Held-out answers never supply training distractors.
        for row in self.data["train"]:
            for slot, value in spans(row):
                if slot in SLOTS:
                    pools[slot].add(value)
        pools = {slot: sorted(values) for slot, values in pools.items()}

        def convert(row: dict) -> list[dict]:
            annotated = spans(row)
            samples = []
            for slot in stable_shuffle(sorted({slot for slot, _ in annotated} & SLOTS.keys()), row["utt"])[:3]:
                values = {value for key, value in annotated if key == slot}
                # Multiple mentions of the same slot can name different valid values. Exclude all of them.
                accepted = {value.casefold() for value in values}
                answer = min(values)
                candidates = [value for value in pools.get(slot, []) if value.casefold() not in accepted]
                seed = row["utt"] + slot
                question = pick_question(QUESTIONS_RU if russian else QUESTIONS, seed).format(slot=SLOTS[slot][russian])
                about = (
                    "Извлеките указанное значение из команды. Используйте слова пользователя, без внешних сведений."
                    if russian
                    else "Extract the specified value from the command, using the user's own words."
                )
                if sample := text_choice(f"{about}\n\n{row['utt']}", question, answer, candidates, seed):
                    samples.append(sample)
            return samples

        self.data = self.data.map(
            explode(convert), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )
        self.data = disjoint_contexts(self.data)


massive_slots_dataset = MassiveSlotsDataset(name="massive_slots", hf_path="AmazonScience/massive", hf_name="en-US")
massive_slots_ru_dataset = MassiveSlotsDataset(
    name="massive_slots_ru", hf_path="AmazonScience/massive", hf_name="ru-RU"
)
