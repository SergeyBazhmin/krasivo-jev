import json
import re
import tarfile

from datasets import Dataset, DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, pick_question, stable_shuffle, text_choice

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
    "date": ("the date, day or calendar period", "дату, день или календарный период"),
    "person": ("the person mentioned", "упомянутого человека"),
    "place_name": ("the place or location", "место или местоположение"),
    "artist_name": ("the artist's name", "имя исполнителя"),
    "song_name": ("the song title", "название песни"),
    "music_album": ("the album title", "название альбома"),
    "podcast_name": ("the podcast title", "название подкаста"),
    "radio_name": ("the radio station or channel requested", "запрошенную радиостанцию или канал"),
    "audiobook_name": ("the audiobook title", "название аудиокниги"),
    "audiobook_author": ("the audiobook author", "автора аудиокниги"),
    "movie_name": ("the movie title", "название фильма"),
    "app_name": ("the application name", "название приложения"),
    "business_name": ("the business name", "название организации"),
    "email_address": ("the email address", "адрес электронной почты"),
    "email_folder": ("the email folder", "папку электронной почты"),
    "event_name": ("the event name or description", "название или описание события"),
    "list_name": ("the list name or description", "название или описание списка"),
    "device_type": ("the device", "устройство"),
    "house_place": ("the room or place in the home", "комнату или место в доме"),
    "color_type": ("the color or shade", "цвет или оттенок"),
    "ingredient": ("the ingredient", "ингредиент"),
    "currency_name": ("the currency", "валюту"),
    "transport_agency": ("the transport operator", "транспортного оператора"),
    "transport_type": ("the means of transport", "вид транспорта"),
}
# playlist_name includes descriptors such as "automated"; time_zone includes ordinary "time"/"day".
# Their source labels do not reliably identify the named entity promised by the question.
ANNOTATION = re.compile(r"\[([^:\[\]]+)\s+:\s+([^\[\]]+)\]")
# These source types can also satisfy the same natural-language question.
RELATED_SLOTS = [
    {"person", "artist_name", "audiobook_author"},
    {"place_name", "house_place", "business_name"},
    {"time", "date", "timeofday", "general_frequency", "time_zone"},
]


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def command_key(text: str) -> str:
    # Source paraphrases often differ only by a vocative or politeness prefix across official splits.
    return re.sub(r"^(?:(?:hey )?(?:alexa|olly|siri|google|assistant)[ ,]+|please )+", "", normalized(text))


def spans(row: dict) -> list[tuple[str, str]]:
    annotated = row["annot_utt"]
    # Reject broken annotations rather than treating the annotation markup as user input.
    if ANNOTATION.sub(lambda match: match[2], annotated) != row["utt"]:
        return []
    return [(match[1].strip(), match[2]) for match in ANNOTATION.finditer(annotated)]


def unique_commands(data: DatasetDict) -> DatasetDict:
    """Deduplicate before expansion; conflicting annotations cannot supply a reliable gold."""
    owners = {}
    annotations = {}
    conflicts = set()
    priority = {"test": 0, "validation": 1, "train": 2}
    for split in sorted(data, key=lambda name: (priority[name], name)):
        for row in data[split]:
            key = command_key(row["utt"])
            signature = tuple(sorted((slot, normalized(value)) for slot, value in spans(row)))
            if key in annotations and annotations[key] != signature:
                conflicts.add(key)
            annotations[key] = signature
            owners.setdefault(key, split)
    result = {}
    for split, rows in data.items():
        seen = set()
        indices = []
        for index, row in enumerate(rows):
            key = command_key(row["utt"])
            if owners[key] == split and key not in conflicts and key not in seen:
                indices.append(index)
                seen.add(key)
        result[split] = rows.select(indices)
    return DatasetDict(result)


def grounded_media(row: dict, russian: bool) -> set[str]:
    """Bare title/creator pairs require cultural knowledge; keep explicit linguistic roles only."""
    supported = set(SLOTS)
    pairs = (("song_name", "artist_name"), ("audiobook_name", "audiobook_author"))
    matches = {match[1].strip(): match for match in ANNOTATION.finditer(row["annot_utt"])}
    for title, creator in pairs:
        present = {title, creator} & matches.keys()
        if not present:
            continue
        creator_marked = False
        if creator in matches:
            prefix = ANNOTATION.sub(lambda match: match[2], row["annot_utt"][: matches[creator].start()])
            marker = r"(?:исполнител\w*|групп\w*|автор\w*)" if russian else r"(?:by|from|artist|band|singer|author)"
            creator_marked = bool(re.search(rf"\b{marker}\s*$", prefix, re.IGNORECASE))
        title_marked = False
        if title in matches:
            prefix = ANNOTATION.sub(lambda match: match[2], row["annot_utt"][: matches[title].start()])
            marker = (
                (r"(?:песн\w*|трек\w*)" if title == "song_name" else r"(?:аудиокниг\w*|книг\w*)")
                if russian
                else (r"(?:song|track)" if title == "song_name" else r"(?:audiobook|book)")
            )
            title_marked = bool(
                re.search(rf"\b{marker}(?:\s+(?:called|named|под названием))?\s*$", prefix, re.IGNORECASE)
            )
            title_marked |= creator_marked and creator in matches and matches[title].end() < matches[creator].start()
        if not creator_marked:
            supported.discard(creator)
        if not title_marked:
            supported.discard(title)
    return supported


def slot_samples(row: dict, russian: bool) -> list[dict]:
    annotated = spans(row)
    supported = grounded_media(row, russian)
    samples = []
    for slot in stable_shuffle(sorted({slot for slot, _ in annotated} & supported), row["utt"]):
        values = {value for key, value in annotated if key == slot}
        # A single-answer question cannot arbitrarily choose one of several currencies, places, etc.
        if len({normalized(value) for value in values}) != 1:
            continue
        answer = min(values)
        accepted = normalized(answer)
        # "date in New York" requests an unknown date rather than supplying a temporal value.
        if slot in {"time", "date"} and accepted in {"time", "date", "время", "дата", "дату"}:
            continue
        related = set().union(*(group for group in RELATED_SLOTS if slot in group)) | {slot}
        # All candidates come from this command. Presence alone must not reveal the answer.
        candidates = sorted(
            {
                value
                for key, value in annotated
                if key in supported
                and key not in related
                and normalized(value) not in accepted
                and accepted not in normalized(value)
            }
        )
        seed = row["utt"] + slot
        question = pick_question(QUESTIONS_RU if russian else QUESTIONS, seed).format(slot=SLOTS[slot][russian])
        about = (
            "Извлеките указанное значение из команды. Используйте слова пользователя, без внешних сведений."
            if russian
            else "Extract the specified value from the command, using the user's own words."
        )
        if sample := text_choice(f"{about}\n\n{row['utt']}", question, answer, candidates, seed):
            samples.append(sample)
        if len(samples) == 3:
            break
    return samples


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
        self.data = unique_commands(self.data)
        self.data = self.data.map(
            explode(lambda row: slot_samples(row, russian)),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


massive_slots_dataset = MassiveSlotsDataset(name="massive_slots", hf_path="AmazonScience/massive", hf_name="en-US")
massive_slots_ru_dataset = MassiveSlotsDataset(
    name="massive_slots_ru", hf_path="AmazonScience/massive", hf_name="ru-RU"
)
