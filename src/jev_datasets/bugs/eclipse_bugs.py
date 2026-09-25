from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

HF_PATH = "AndressaStefany/bug-reports"

SEVERITIES = ["blocker", "critical", "major", "normal", "minor", "trivial", "enhancement"]
# the rare MOVED and the undocumented NDUPLICATE are dropped
RESOLUTIONS = {
    "FIXED": "fixed",
    "DUPLICATE": "duplicate of another report",
    "WONTFIX": "won't fix",
    "WORKSFORME": "cannot reproduce",
    "INVALID": "invalid",
    "NOT_ECLIPSE": "not an Eclipse problem",
}


class EclipseBugsDataset(JevDataset):
    """Eclipse Bugzilla reports; one label column predicted from the report's description."""

    def __init__(self, name: str, column: str, options: dict[str, str], question: str):
        super().__init__(name=name, hf_path=HF_PATH)
        self.column = column
        self.options = make_options(list(options), list(options.values()))
        self.question = question

    def prepare(self):
        ids = {option["id"] for option in self.options}
        self.data = self.data.filter(lambda x: x["description"] and x[self.column] in ids)
        self.data = self.data.map(
            lambda x: make_sample(x["description"], self.question, self.options, x[self.column]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


eclipse_bug_severity_dataset = EclipseBugsDataset(
    name="eclipse_bug_severity",
    column="bug_severity",
    options={severity: severity for severity in SEVERITIES},
    question="What severity should this bug report be filed with?",
)
eclipse_bug_resolution_dataset = EclipseBugsDataset(
    name="eclipse_bug_resolution",
    column="resolution",
    options=RESOLUTIONS,
    question="How will this bug report most likely be resolved?",
)
