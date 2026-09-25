from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify

QUESTIONS = [
    "What is the topic of this financial news tweet?",
    "Which financial news category does this tweet fall under?",
    "What is this finance tweet about?",
    "Which topic best describes this market news post?",
    "How would you categorize this financial tweet?",
]

# `label` is a bare int; the names come from the dataset card
TOPICS = [
    "Analyst Update",
    "Fed | Central Banks",
    "Company | Product News",
    "Treasuries | Corporate Debt",
    "Dividend",
    "Earnings",
    "Energy | Oil",
    "Financials",
    "Currencies",
    "General News | Opinion",
    "Gold | Metals | Materials",
    "IPO",
    "Legal | Regulation",
    "M&A | Investments",
    "Macro",
    "Markets",
    "Politics",
    "Personnel Change",
    "Stock Commentary",
    "Stock Movement",
]
IDS = [slugify(topic) for topic in TOPICS]
OPTIONS = make_options(IDS, TOPICS)


class FinancialNewsTopicDataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(x["text"], pick_question(QUESTIONS, x["text"]), OPTIONS, IDS[x["label"]]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


financial_news_topic_dataset = FinancialNewsTopicDataset(
    name="financial_news_topic", hf_path="zeroshot/twitter-financial-news-topic"
)
