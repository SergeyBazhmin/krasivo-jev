from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What type of answer does this question ask for?",
    "Which kind of answer is the question looking for?",
    "What would the answer to this question be?",
    "Which answer type does this question expect?",
    "Classify the question by the type of its answer.",
]
ABOUT = [
    "You are given a question. Decide what type of answer it asks for. Do not answer the question itself.",
    "Below is a question. Without answering it, decide which type of answer it expects.",
    "Classify the question below by the type of answer it asks for. You do not need to know the answer.",
    "You sort questions by the kind of answer they expect. Decide the answer type of the question below.",
    "A question is given. Decide what kind of answer would satisfy it, without answering it.",
]
# The fine taxonomy of Li & Roth (2002). Option texts must be unique, so the "desc" and "other" labels name their
# coarse class. Some labels turn on what the asked-about thing is ("What is Mars?" is a definition, "What is Mars
# made of?" a substance), so the guide spells out every label.
FINE = {
    "ABBR:abb": ("abbreviation", "the abbreviated form of something"),
    "ABBR:exp": ("expansion of an abbreviation", "what an abbreviation stands for"),
    "ENTY:animal": ("animal", "an animal"),
    "ENTY:body": ("body part", "an organ or part of the body"),
    "ENTY:color": ("color", "a color"),
    "ENTY:cremat": ("creative work", "a book, film, song, painting, invention or other creative piece"),
    "ENTY:currency": ("currency", "the name of a currency"),
    "ENTY:dismed": ("disease or medicine", "a disease or a medicine"),
    "ENTY:event": ("event", "an event"),
    "ENTY:food": ("food", "a food or drink"),
    "ENTY:instru": ("musical instrument", "a musical instrument"),
    "ENTY:lang": ("language", "a language"),
    "ENTY:letter": ("letter", "a letter, such as a to z"),
    "ENTY:other": ("other entity", "an entity of a kind not listed here"),
    "ENTY:plant": ("plant", "a plant"),
    "ENTY:product": ("product", "a product"),
    "ENTY:religion": ("religion", "a religion"),
    "ENTY:sport": ("sport", "a sport or game"),
    "ENTY:substance": ("substance", "an element, material or substance"),
    "ENTY:symbol": ("symbol", "a symbol or sign"),
    "ENTY:techmeth": ("technique or method", "a technique or method"),
    "ENTY:termeq": ("equivalent term", "another term for something described"),
    "ENTY:veh": ("vehicle", "a vehicle"),
    "ENTY:word": ("word", "a word with a special property"),
    "DESC:def": ("definition", "the definition of something"),
    "DESC:desc": ("description of something", "a description of something"),
    "DESC:manner": ("manner", "how something is done"),
    "DESC:reason": ("reason", "why something happens"),
    "HUM:gr": ("group", "a group or organization of people"),
    "HUM:ind": ("individual", "a single person"),
    "HUM:title": ("title", "the title or position of a person"),
    "HUM:desc": ("description of a person", "a description of a person"),
    "LOC:city": ("city", "a city"),
    "LOC:country": ("country", "a country"),
    "LOC:mount": ("mountain", "a mountain"),
    "LOC:other": ("other location", "a location of a kind not listed here"),
    "LOC:state": ("state", "a state or province"),
    "NUM:code": ("code", "a postcode, phone number or other code"),
    "NUM:count": ("count", "how many of something"),
    "NUM:date": ("date", "a date or year"),
    "NUM:dist": ("distance", "a length, height or other linear measure"),
    "NUM:money": ("money", "a price or amount of money"),
    "NUM:ord": ("rank", "a position in an order"),
    "NUM:other": ("other number", "a number of a kind not listed here"),
    "NUM:period": ("period", "how long something lasts, or an age"),
    "NUM:perc": ("percentage", "a percentage or fraction"),
    "NUM:speed": ("speed", "a speed"),
    "NUM:temp": ("temperature", "a temperature"),
    "NUM:volsize": ("size", "a size, area or volume"),
    "NUM:weight": ("weight", "a weight"),
}


class TrecDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # the main branch is a loading script, which `datasets` no longer runs; the Hub's parquet export
        return load_dataset(self.hf_path, revision="refs/convert/parquet")

    def prepare(self):
        names = self.class_names("fine_label")
        options = make_options(list(FINE), [text for text, _ in FINE.values()])
        answers = dict(FINE.values())
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Question", answers),
                pick_question(QUESTIONS, x["text"]),
                options,
                names[x["fine_label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


trec_dataset = TrecDataset(name="trec", hf_path="CogComp/trec")
