from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "What kind of GitHub issue is this?"

# The other labels (Duplicate, Invalid, Help Wanted, Good First Issue) describe triage
# state rather than what the issue is about, so those issues are dropped.
KINDS = {"Bug": "bug", "Enhancement": "enhancement", "Question": "question", "Documentation": "documentation"}
OPTIONS = make_options(list(KINDS.values()))


class GitHubIssueTypeDataset(JevDataset):
    def prepare(self):
        # text is pre-lowercased with punctuation stripped by the source
        self.data = self.data.filter(lambda x: x["labels"] in KINDS)
        self.data = self.data.map(
            lambda x: make_sample(
                "\n\n".join(part.strip() for part in (x["issue title"], x["body"]) if part and part.strip()),
                QUESTION,
                OPTIONS,
                KINDS[x["labels"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


github_issue_type_dataset = GitHubIssueTypeDataset(
    name="github_issue_type", hf_path="khanmu2003/issue-classification"
)
