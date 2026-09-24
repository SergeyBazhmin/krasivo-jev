from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class TweetModerationDataset(JevDataset):
    def __init__(self, name: str, hf_name: str, question: str):
        super().__init__(name=name, hf_path="cardiffnlp/tweet_eval", hf_name=hf_name)
        self.question = question

    def prepare(self):
        # e.g. "non-offensive" / "offensive", "non-hate" / "hate"
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], self.question, options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


tweet_offensive_dataset = TweetModerationDataset(
    name="tweet_offensive", hf_name="offensive", question="Is this tweet offensive?"
)
tweet_hate_dataset = TweetModerationDataset(
    name="tweet_hate", hf_name="hate", question="Is this tweet hate speech?"
)
