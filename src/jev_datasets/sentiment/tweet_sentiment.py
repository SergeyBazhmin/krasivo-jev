from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is the sentiment of this tweet?",
    "Is this tweet positive, negative or neutral?",
    "What is the overall tone of this tweet?",
    "How does the author of this tweet feel?",
    "What attitude does this tweet express?",
]
ABOUT = (
    "You are given a tweet, with user mentions replaced by @user. Decide whether its author expresses a positive or a "
    "negative attitude, or neither (neutral), as in plain news or factual statements."
)


class TweetSentimentDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # negative, neutral, positive
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Tweet"), pick_question(QUESTIONS, x["text"]), options, names[x["label"]]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


tweet_sentiment_dataset = TweetSentimentDataset(
    name="tweet_sentiment", hf_path="cardiffnlp/tweet_eval", hf_name="sentiment"
)
