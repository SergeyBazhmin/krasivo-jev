from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "What kind of GitHub issue is this?",
    "How should this GitHub issue be labeled?",
    "Is this issue a bug, an enhancement, a question or about documentation?",
    "What type of issue did the author open?",
    "Which label best describes this issue?",
]

# The other labels (Duplicate, Invalid, Help Wanted, Good First Issue) describe triage
# state rather than what the issue is about, so those issues are dropped.
KINDS = {"Bug": "bug", "Enhancement": "enhancement", "Question": "question", "Documentation": "documentation"}
OPTIONS = make_options(list(KINDS.values()))


class GitHubIssueTypeDataset(JevDataset):
    def prepare(self):
        # text is pre-lowercased with punctuation stripped by the source
        self.data = self.data.filter(lambda x: x["labels"] in KINDS)
        self.data = self.data.map(
            lambda x: self.to_sample(x),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )

    @staticmethod
    def to_sample(x: dict) -> dict:
        issue = "\n\n".join(part.strip() for part in (x["issue title"], x["body"]) if part and part.strip())
        return make_sample(issue, pick_question(QUESTIONS, issue), OPTIONS, KINDS[x["labels"]])


github_issue_type_dataset = GitHubIssueTypeDataset(
    name="github_issue_type", hf_path="khanmu2003/issue-classification"
)
