from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, slugify

QUESTION = "Which topic does this news article belong to?"


class AGNewsDataset(JevDataset):
    def prepare(self):
        names = self.class_names()
        ids = [slugify(name) for name in names]  # "Sci/Tech" -> "sci_tech"
        options = make_options(ids, names)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, options, ids[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ag_news_dataset = AGNewsDataset(name="ag_news", hf_path="fancyzhx/ag_news")
