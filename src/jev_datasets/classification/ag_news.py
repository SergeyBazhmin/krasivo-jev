from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify

QUESTIONS = [
    "Which topic does this news article belong to?",
    "What is this news article about?",
    "Which news section would this article appear in?",
    "What category best fits this news story?",
    "Classify the topic of this article.",
]


class AGNewsDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()
        ids = [slugify(name) for name in names]  # "Sci/Tech" -> "sci_tech"
        options = make_options(ids, names)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], pick_question(QUESTIONS, x["text"]), options, ids[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ag_news_dataset = AGNewsDataset(name="ag_news", hf_path="fancyzhx/ag_news")
