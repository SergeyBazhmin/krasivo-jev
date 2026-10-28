from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import GROUPED_FEATURES, make_options, make_sample, pick_question, with_context

# tweet_eval's stance tasks are left out: their tweets often imply the stance only through outside knowledge (a
# Mother Teresa quote labelled as against abortion). Here a claim states its argument for or against the motion.
QUESTIONS = [
    "Does the claim support or oppose the motion?",
    "Is this claim an argument for or against the motion?",
    "Which side of the debate does the claim take?",
    "Would a speaker use this claim to argue for or against the motion?",
    "What is the claim's stance towards the motion?",
]
OPTIONS = make_options(["PRO", "CON"], ["supports the motion", "opposes the motion"])
ABOUT = [
    (
        "You are given a debate motion and a claim taken from an article on its topic. Decide whether the claim "
        "argues for the motion or against it."
    ),
    (
        "Below are a debate motion and a claim about its topic. Decide which side the claim takes: does it support "
        "the motion or oppose it?"
    ),
    (
        "A debate is held on the motion below. Determine whether the given claim is an argument in favour of the "
        "motion or against it."
    ),
    (
        "You judge the stance of claims in a debate. Given the motion and one claim, decide whether the claim backs "
        "the motion or argues against it."
    ),
    (
        "Read the debate motion and the claim. A claim supports the motion when it argues that the motion should be "
        "accepted, and opposes it when it argues that the motion should be rejected."
    ),
]


class ClaimStanceDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        def convert(row: dict) -> dict:
            debate = f"Motion: {row['topicText'].strip()}\nClaim: {row['claims.claimCorrectedText'].strip()}"
            sample = make_sample(
                with_context(ABOUT, debate), pick_question(QUESTIONS, debate), OPTIONS, row["claims.stance"]
            )
            # Keep each motion's claims in one split, so held-out motions are unseen.
            return {**sample, "group": str(row["topicId"])}

        self.data = self.data.filter(
            lambda x: x["claims.stance"] in ("PRO", "CON") and bool(x["claims.claimCorrectedText"].strip())
        )
        self.data = self.data.map(convert, remove_columns=self.source_columns, features=GROUPED_FEATURES)


claim_stance_dataset = ClaimStanceDataset(
    name="claim_stance", hf_path="ibm-research/claim_stance", hf_name="claim_stance"
)
