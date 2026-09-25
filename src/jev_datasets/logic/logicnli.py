import re

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question

QUESTIONS = [
    "Given the facts and rules, is the hypothesis entailed, contradicted, neutral, or a paradox?",
    "Does the hypothesis follow from the premise, does its negation, both, or neither?",
    "What is the logical relation between the premise and the hypothesis?",
    "Based only on the facts and rules, what can be concluded about the hypothesis?",
    "Can the hypothesis, its negation, both, or neither be derived from the premise?",
]
# "self_contradiction" is the paper's "paradox": both the hypothesis and its negation follow
OPTIONS = make_options(
    ["entailment", "contradiction", "neutral", "self_contradiction"],
    ["entailment", "contradiction", "neutral", "paradox"],
)
# the source glues the rules onto the last fact with no separator: "... not accurate.If there ..."
GLUED = re.compile(r"(?<=[a-z]\.)(?=[A-Z])")


class LogicNLIDataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(
                GLUED.sub("\n", x["premise"]),
                f"Hypothesis: {x['hypothesis']}\n{pick_question(QUESTIONS, x['premise'] + x['hypothesis'])}",
                OPTIONS,
                x["label"],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


logicnli_dataset = LogicNLIDataset(name="logicnli", hf_path="tasksource/LogicNLI")
