from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "Is this email spam?",
    "Should this email go to the spam folder?",
    "Is this an unsolicited bulk email rather than a work or personal message?",
    "Would a spam filter be right to block this email?",
    "Is this message spam?",
]
OPTIONS = make_options(["yes", "no"])
ABOUT = [
    (
        "You filter a company employee's inbox. Spam is unsolicited bulk email: ads for cheap software, drugs or "
        "stocks, get-rich offers, adult content and similar mass mailings. Business correspondence, newsletters the "
        "employee works with, and personal mail are not spam. The text is lowercased and tokenized."
    ),
    (
        "Below is an email from an employee's mailbox, lowercased and tokenized. Decide whether it is spam, meaning "
        "unsolicited bulk mail such as ads for software, pharmaceuticals or stocks, money-making offers or adult "
        "content. Work emails and personal messages are not spam."
    ),
    (
        "You are a spam filter for a corporate mailbox. Unsolicited mass mailings, such as advertising for cheap "
        "software, medication or stock tips, offers to get rich and adult content, are spam; business and personal "
        "correspondence is not. The email is lowercased and tokenized."
    ),
    (
        "Classify the email below, taken from an employee's inbox and lowercased and tokenized. Spam is unsolicited "
        "bulk advertising: discount software, pills, stock promotions, easy-money schemes, adult content. Ordinary "
        "work and personal email is not spam."
    ),
    (
        "Decide whether this email from a corporate inbox is spam. Spam means unsolicited mass mail, for example "
        "ads for software, drugs or stocks, money-making schemes or adult material. Messages about work or from "
        "people the employee knows are not spam. The text is lowercased and tokenized."
    ),
]


def email(row: dict) -> str:
    return f"Subject: {row['subject'].strip()}\n\n{row['message'].strip()}"


class EnronSpamDataset(JevDataset):
    type = DatasetType.NOUL

    def prepare(self):
        self.data = self.data.filter(lambda x: bool(x["subject"].strip() or x["message"].strip()))
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, email(x), "Email"),
                pick_question(QUESTIONS, email(x)),
                OPTIONS,
                "yes" if x["label_text"] == "spam" else "no",
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


enron_spam_dataset = EnronSpamDataset(name="enron_spam", hf_path="SetFit/enron_spam")
