from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample


class SpaRPDataset(JevDataset):
    def prepare(self):
        # A third of the questions have several relations that all hold ("below" and "behind");
        # one distribution over the options can't say "all of these", so only single-answer
        # questions are kept. Those flagged as needing commonsense are dropped too.
        self.data = self.data.filter(lambda x: sum(x["target_scores"]) == 1 and not x["commonsense_question"])
        self.data = self.data.map(
            lambda x: make_sample(x["context"], x["question"], make_options(x["target_choices"]), x["targets"][0]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


# only the SpaRTUN part: the other configs are StepGame and its extensions, already in `stepgame`
sparp_dataset = SpaRPDataset(name="sparp", hf_path="UKPLab/sparp", hf_name="SpaRP-PS1 (SpaRTUN)")
