from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "What is the sentiment of this tweet?",
    "Is this tweet positive, negative or neutral?",
    "What is the overall tone of this tweet?",
    "How does the author of this tweet feel?",
    "What attitude does this tweet express?",
]


class TweetSentimentDataset(JevDataset):
    def prepare(self):
        # negative, neutral, positive
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(x["text"], pick_question(QUESTIONS, x["text"]), options, names[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


tweet_sentiment_dataset = TweetSentimentDataset(
    name="tweet_sentiment", hf_path="cardiffnlp/tweet_eval", hf_name="sentiment"
)
