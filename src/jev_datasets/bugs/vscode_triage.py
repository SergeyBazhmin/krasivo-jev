from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Is this GitHub issue a bug report or a feature request?",
    "Does this issue report a bug or request a feature?",
    "Should this issue be triaged as a bug or as a feature request?",
    "Is the author reporting a defect or asking for new functionality?",
    "How should this VS Code issue be labeled?",
]
OPTIONS = make_options(["bug", "feature-request"], ["bug", "feature request"])


class VSCodeTriageDataset(JevDataset):
    def prepare(self):
        # `input` is already "Title: ...\n\nDescription: ..."
        self.data = self.data.map(
            lambda x: make_sample(x["input"], pick_question(QUESTIONS, x["input"]), OPTIONS, x["label"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


vscode_triage_dataset = VSCodeTriageDataset(name="vscode_triage", hf_path="Tilakoid/vscode-bug-feature-triage")
