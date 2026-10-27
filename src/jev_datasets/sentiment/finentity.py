from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample, pick_question, with_context

QUESTIONS = [
    "What is the sentiment toward this entity?",
    "How does the text portray this entity?",
    "Is the news positive, neutral or negative for this entity?",
    "Which sentiment does the text express about the entity?",
    "How is this entity affected, according to the text?",
]
ABOUT = [
    (
        "You analyse financial news at the level of entities. Below is a news snippet; for the named company, "
        "asset or other financial entity, decide whether the text is positive, neutral or negative about it. "
        "Another entity in the same text can get a different sentiment."
    ),
    (
        "Below is a piece of financial news. Judge the sentiment it expresses toward one entity named in it: "
        "positive, neutral or negative. Different entities in one text can have different sentiments."
    ),
    (
        "You read financial news and rate how it reflects on one of the companies, assets or institutions it "
        "mentions. The rating is about that entity alone, not the text as a whole."
    ),
    (
        "As a financial analyst, you judge what a news snippet means for a specific entity in it. Only the "
        "sentiment toward that entity counts; other entities in the snippet may fare differently."
    ),
    (
        "A financial news text follows. For the entity in question, say whether the text is good news for it, "
        "bad news, or neither; each entity in the text is judged on its own."
    ),
]
ANSWERS = {
    "positive": "the text reports gains, growth, success or other favourable news for the entity",
    "neutral": "the text mentions the entity without a favourable or unfavourable slant",
    "negative": "the text reports losses, declines, problems or other unfavourable news for the entity",
}
OPTIONS = make_options(list(ANSWERS))


def finentity_samples(x: dict) -> list[dict]:
    labels: dict[str, set[str]] = {}
    for annotation in x["annotations"]:
        labels.setdefault(annotation["value"], set()).add(annotation["label"].lower())
    state = with_context(ABOUT, x["content"].strip(), "News", ANSWERS)
    return [
        make_sample(state, f"Entity: {entity}\n{pick_question(QUESTIONS, state + entity)}", OPTIONS, next(iter(label)))
        for entity, label in labels.items()
        # an entity mentioned twice with different sentiments has no single answer
        if len(label) == 1 and next(iter(label)) in ANSWERS
    ]


class FinEntityDataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = self.data.map(
            explode(finentity_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


finentity_dataset = FinEntityDataset(name="finentity", hf_path="yixuantt/FinEntity")
