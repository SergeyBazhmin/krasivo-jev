from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify, with_context

QUESTIONS = [
    "Which topic does this news article belong to?",
    "What is this news article about?",
    "Which news section would this article appear in?",
    "What category best fits this news story?",
    "Classify the topic of this article.",
]
ABOUT = [
    (
        "You are given the title and description of a news article. Decide which section of a news site it belongs "
        "in: World (international news and politics), Sports, Business, or Sci/Tech (science and technology)."
    ),
    (
        "You are given a news article's title and description. Decide which news site section it belongs in: World "
        "(international news and politics), Sports, Business, or Sci/Tech (science and technology)."
    ),
    (
        "Below are the title and description of a news article. Sort it into one section of a news site: World for "
        "international news and politics, Sports, Business, or Sci/Tech for science and technology."
    ),
    (
        "You are an editor filing articles into the sections of a news site. Given an article's title and "
        "description, decide whether it goes under World (international news and politics), Sports, Business, or "
        "Sci/Tech (science and technology)."
    ),
    (
        "A news article is given by its title and description. Pick the section of a news site where it belongs: "
        "World (international affairs and politics), Sports, Business, or Sci/Tech (science and technology)."
    ),
]


class AGNewsDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        names = self.class_names()
        ids = [slugify(name) for name in names]  # "Sci/Tech" -> "sci_tech"
        options = make_options(ids, names)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Article"), pick_question(QUESTIONS, x["text"]), options, ids[x["label"]]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


ag_news_dataset = AGNewsDataset(name="ag_news", hf_path="fancyzhx/ag_news")
