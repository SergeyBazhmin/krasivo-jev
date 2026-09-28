from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, slugify, with_context

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
ABOUT = (
    "You are given a tweet from a financial news account on Twitter. Decide which topic it covers, judging by what it "
    "reports: for example an analyst's rating change, a central bank decision, a company's earnings, a dividend, a "
    "merger or acquisition, or a move in a stock's price (as opposed to commentary or opinion about a stock)."
)
IDS = [slugify(topic) for topic in TOPICS]
OPTIONS = make_options(IDS, TOPICS)


class FinancialNewsTopicDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["text"], "Tweet"), pick_question(QUESTIONS, x["text"]), OPTIONS, IDS[x["label"]]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


financial_news_topic_dataset = FinancialNewsTopicDataset(
    name="financial_news_topic", hf_path="zeroshot/twitter-financial-news-topic"
)
