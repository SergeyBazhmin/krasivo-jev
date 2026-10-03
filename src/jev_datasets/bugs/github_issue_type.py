from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.bugs.github import GITHUB_INTRO
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

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
LABELS = {
    "bug": "something in the software does not work as it should (an error, a crash, wrong behavior)",
    "enhancement": "a request for a new feature or an improvement to how the software works",
    "question": "the author asks how to do something or how the software works, without reporting a defect",
    "documentation": "a problem with or a request about the docs, README, comments or examples, not the code",
}


class GitHubIssueTypeDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # text is pre-lowercased with punctuation stripped by the source
        self.data = self.data.filter(lambda x: x["labels"] in KINDS)
        self.data = self.data.map(
            lambda x: self.to_sample(x),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
            num_proc=0,
        )

    @staticmethod
    def to_sample(x: dict) -> dict:
        issue = "\n\n".join(part.strip() for part in (x["issue title"], x["body"]) if part and part.strip())
        state = with_context(GITHUB_INTRO, issue, "Issue", LABELS)
        return make_sample(state, pick_question(QUESTIONS, issue), OPTIONS, KINDS[x["labels"]])


github_issue_type_dataset = GitHubIssueTypeDataset(name="github_issue_type", hf_path="khanmu2003/issue-classification")
