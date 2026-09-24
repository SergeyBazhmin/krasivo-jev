from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "What is the sentiment of this tweet?"


class TweetSentimentDataset(JevDataset):
    def prepare(self):
        # negative, neutral, positive
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], QUESTION, options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


tweet_sentiment_dataset = TweetSentimentDataset(
    name="tweet_sentiment", hf_path="cardiffnlp/tweet_eval", hf_name="sentiment"
)
