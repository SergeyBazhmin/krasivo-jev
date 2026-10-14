from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, with_context

ABOUT = [
    (
        "You are given a description of boxes (or blocks) holding objects of different sizes and colors, and of "
        "where they are relative to each other. Compass directions describe the picture: north is above, south "
        "below, west left and east right. Work out the relation asked about with spatial logic: relations have "
        "inverses (if A is left of B, B is right of A), they chain, and what is inside a box shares that box's "
        "position relative to other boxes."
    ),
    (
        "Below is a description of boxes (or blocks) holding objects of different sizes and colors, and of where "
        "they are relative to each other. Compass directions describe the picture: north is above, south below, "
        "west left and east right. Use spatial logic to work out the relation asked about: relations have inverses "
        "(if A is left of B, B is right of A), they chain, and what is inside a box shares that box's position "
        "relative to other boxes."
    ),
    (
        "You are given a scene of boxes (or blocks) containing objects of various sizes and colors, described by "
        "where things are relative to each other. In the picture north is above, south below, west to the left and "
        "east to the right. Find the relation asked about by spatial logic: every relation has an inverse (A left "
        "of B means B right of A), relations chain, and objects inside a box share its position relative to other "
        "boxes."
    ),
    (
        "You reason about spatial relations in a scene of boxes (or blocks) that hold objects of different sizes "
        "and colors. Compass directions map onto the picture: north above, south below, west left, east right. "
        "Relations have inverses (if A is left of B, B is right of A) and chain, and anything inside a box takes "
        "that box's position relative to other boxes. Work out the relation asked about."
    ),
    (
        "A scene describes boxes (or blocks) with objects of different sizes and colors in them, and their "
        "positions relative to each other. North means above, south below, west left and east right. Work out the "
        "asked relation with spatial logic: relations invert (if A is left of B, B is right of A), they chain, and "
        "what a box contains shares the box's position relative to other boxes."
    ),
]


class SpaRPDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        # A third of the questions have several relations that all hold ("below" and "behind");
        # one distribution over the options can't say "all of these", so only single-answer
        # questions are kept. Those flagged as needing commonsense are dropped too.
        self.data = self.data.filter(lambda x: sum(x["target_scores"]) == 1 and not x["commonsense_question"])
        self.data = self.data.map(
            lambda x: make_sample(
                with_context(ABOUT, x["context"], "Story"),
                x["question"],
                make_options(x["target_choices"]),
                x["targets"][0],
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


# only the SpaRTUN part: the other configs are StepGame and its extensions, already in `stepgame`
sparp_dataset = SpaRPDataset(name="sparp", hf_path="UKPLab/sparp", hf_name="SpaRP-PS1 (SpaRTUN)")
