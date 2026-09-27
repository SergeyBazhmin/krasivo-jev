from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify

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


class DBpediaDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()
        ids = [slugify(TEXTS[name]) for name in names]
        options = make_options(ids, [TEXTS[name] for name in names])
        self.data = self.data.map(
            lambda x: make_sample(
                (text := f"{x['title']}\n\n{x['content'].strip()}"),
                pick_question(QUESTIONS, text),
                options,
                ids[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


dbpedia_dataset = DBpediaDataset(name="dbpedia", hf_path="fancyzhx/dbpedia_14")
