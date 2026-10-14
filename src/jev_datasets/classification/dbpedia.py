from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify, with_context

QUESTIONS = [
    "What kind of entity is described in this article?",
    "Which category does the subject of this text belong to?",
    "What is this encyclopedia entry about?",
    "Which type best fits the entity described here?",
    "Classify the subject of this Wikipedia abstract.",
]

# source class names are CamelCase ontology types
TEXTS = {
    "Company": "company",
    "EducationalInstitution": "educational institution",
    "Artist": "artist",
    "Athlete": "athlete",
    "OfficeHolder": "office holder",
    "MeanOfTransportation": "means of transportation",
    "Building": "building",
    "NaturalPlace": "natural place",
    "Village": "village",
    "Animal": "animal",
    "Plant": "plant",
    "Album": "album",
    "Film": "film",
    "WrittenWork": "written work",
}
ABOUT = [
    (
        "You are given the title and opening of an encyclopedia article. Decide what kind of thing its subject is, "
        "such as a company, an artist, an athlete, a village, an animal, an album or a film."
    ),
    (
        "Below are the title and opening of an encyclopedia article. Decide what kind of thing the article's "
        "subject is, for example a company, an artist, an athlete, a village, an animal, an album or a film."
    ),
    (
        "You are given an encyclopedia article's title and first lines. Classify its subject by kind: a company, "
        "an artist, an athlete, a village, an animal, an album, a film, and so on."
    ),
    (
        "You sort encyclopedia articles by the type of their subject. Given an article's title and opening, decide "
        "whether the subject is, for instance, a company, an artist, an athlete, a village, an animal, an album or "
        "a film."
    ),
    (
        "An encyclopedia article is given by its title and its opening text. Decide which kind of thing it is "
        "about, such as a company, an artist, an athlete, a village, an animal, an album or a film."
    ),
]


class DBpediaDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()
        ids = [slugify(TEXTS[name]) for name in names]
        options = make_options(ids, [TEXTS[name] for name in names])
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, text := f"{x['title']}\n\n{x['content'].strip()}", "Article"),
                pick_question(QUESTIONS, text),
                options,
                ids[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


dbpedia_dataset = DBpediaDataset(name="dbpedia", hf_path="fancyzhx/dbpedia_14")
