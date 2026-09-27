from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.bugs.github import with_github_context
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Is this GitHub issue a bug report or a feature request?",
    "Does this issue report a bug or request a feature?",
    "Should this issue be triaged as a bug or as a feature request?",
    "Is the author reporting a defect or asking for new functionality?",
    "How should this VS Code issue be labeled?",
]
OPTIONS = make_options(["bug", "feature-request"], ["bug", "feature request"])
LABELS = {
    "bug": "something in VS Code, a code editor, does not work as it should (an error, a crash, wrong behavior)",
    "feature request": "a request for new functionality or a change to how VS Code currently works",
}


class VSCodeTriageDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # `input` is already "Title: ...\n\nDescription: ..."
        self.data = self.data.map(
            lambda x: make_sample(
                with_github_context(x["input"], LABELS), pick_question(QUESTIONS, x["input"]), OPTIONS, x["label"]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


vscode_triage_dataset = VSCodeTriageDataset(name="vscode_triage", hf_path="Tilakoid/vscode-bug-feature-triage")
