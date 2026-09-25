from string import ascii_uppercase

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

# SpartQA-Auto: synthetic block scenes; its labels are known to be somewhat noisy
YN_OPTIONS = make_options(["Yes", "No", "DK"], ["yes", "no", "cannot be determined"])


class SpartQAMultipleChoiceDataset(JevDataset):
    def prepare(self):
        # candidates are the two objects the question names, then "both of them" and "none of them"
        self.data = self.data.map(
            lambda x: make_sample(
                x["story"],
                x["question"],
                make_options(
                    list(ascii_uppercase[: len(x["candidate_answers"])]),
                    [" ".join(answer.split()) for answer in x["candidate_answers"]],
                ),
                ascii_uppercase[x["answer"]],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


class SpartQAYesNoDataset(JevDataset):
    def prepare(self):
        self.data = self.data.map(
            lambda x: make_sample(x["story"], x["question"], YN_OPTIONS, x["answer"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


spartqa_mchoice_dataset = SpartQAMultipleChoiceDataset(name="spartqa_mchoice", hf_path="tasksource/spartqa-mchoice")
spartqa_yn_dataset = SpartQAYesNoDataset(name="spartqa_yn", hf_path="tasksource/spartqa-yn")
