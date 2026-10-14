from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context


class TweetModerationDataset(JevDataset):
    type = DatasetType.CHOICE

    def __init__(self, name: str, hf_name: str, about: list[str], questions: list[str]):
        super().__init__(name=name, hf_path="cardiffnlp/tweet_eval", hf_name=hf_name)
        self.about = about
        self.questions = questions

    def prepare(self):
        # e.g. "non-offensive" / "offensive", "non-hate" / "hate"
        names = self.class_names()
        options = make_options(names)
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(self.about, x["text"], "Tweet"),
                pick_question(self.questions, x["text"]),
                options,
                names[x["label"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


tweet_offensive_dataset = TweetModerationDataset(
    name="tweet_offensive",
    hf_name="offensive",
    about=[
        (
            "You are a content moderator for Twitter. You are given a tweet, with user mentions replaced by @user. "
            "A tweet is offensive when it contains insults, threats, profanity or any other offensive language, "
            "whether aimed at a person, a group or no one in particular."
        ),
        (
            "You moderate content on Twitter. Below is a tweet with user mentions replaced by @user. It is "
            "offensive if it contains insults, threats, profanity or any other offensive language, whether aimed "
            "at a person, a group or no one in particular."
        ),
        (
            "Below is a tweet in which user mentions are replaced by @user. As a Twitter content moderator, decide "
            "whether it is offensive: a tweet is offensive when it has insults, threats, profanity or other "
            "offensive language, aimed at a person, a group or nobody in particular."
        ),
        (
            "As a content moderator for Twitter, you flag offensive tweets (user mentions appear as @user). A "
            "tweet is offensive if it contains insults, threats, profanity or any other offensive language, aimed "
            "at someone, at a group, or at no one in particular."
        ),
        (
            "A tweet is shown below with user mentions replaced by @user. Offensive tweets contain insults, "
            "threats, profanity or other offensive language, whether directed at a person, a group or no one in "
            "particular."
        ),
    ],
    questions=[
        "Is this tweet offensive?",
        "Does this tweet contain offensive language?",
        "Would this tweet offend people?",
        "Should this tweet be flagged as offensive?",
        "Is the language in this tweet offensive?",
    ],
)
tweet_hate_dataset = TweetModerationDataset(
    name="tweet_hate",
    hf_name="hate",
    about=[
        (
            "You are a content moderator for Twitter. You are given a tweet, with user mentions replaced by @user. "
            "A tweet is hate speech when it attacks or incites hatred against immigrants or women because they "
            "belong to that group; being rude or negative is not enough on its own."
        ),
        (
            "You moderate content on Twitter. Below is a tweet with user mentions replaced by @user. It is hate "
            "speech if it attacks or incites hatred against immigrants or women because they belong to that group; "
            "being rude or negative is not enough by itself."
        ),
        (
            "Below is a tweet in which user mentions are replaced by @user. As a Twitter content moderator, decide "
            "whether it is hate speech: a tweet that attacks or stirs up hatred against immigrants or women for "
            "belonging to that group. Rudeness or negativity alone does not count."
        ),
        (
            "As a content moderator for Twitter, you flag hate speech in tweets (user mentions appear as @user). "
            "Hate speech attacks or incites hatred against immigrants or women because they are part of that "
            "group; a tweet that is merely rude or negative is not hate speech."
        ),
        (
            "A tweet is shown below with user mentions replaced by @user. It is hate speech when it attacks or "
            "incites hatred against immigrants or women because they belong to that group, not merely when it is "
            "rude or negative."
        ),
    ],
    questions=[
        "Is this tweet hate speech?",
        "Does this tweet express hate against a group of people?",
        "Does this tweet attack people for who they are?",
        "Should this tweet be flagged as hate speech?",
        "Is this tweet hateful towards immigrants or women?",
    ],
)
