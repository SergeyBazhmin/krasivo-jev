from collections import defaultdict
from string import ascii_uppercase

from datasets import Dataset, DatasetDict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.constants import MAX_DISTRACTORS
from jev_datasets.utils import (
    GROUPED_FEATURES,
    SAMPLE_FEATURES,
    explode,
    make_options,
    make_sample,
    pick_about,
    pick_question,
    stable_shuffle,
)

QUESTIONS = [
    "Which passage answers the query?",
    "Select the passage that provides an answer to the query.",
    "Which of these passages contains the requested information?",
    "Choose the passage that answers the supplied question.",
    "In which passage can the query be answered?",
]
ABOUT = [
    (
        "Each option is a passage. Select the passage that answers the query, using only the supplied text. Choose "
        "'None of these passages' if none answers the query."
    ),
    (
        "Each option is a passage. Pick the one that answers the query, using only the supplied text, or choose "
        "'None of these passages' if none of them answers it."
    ),
    (
        "The options are passages. Using only their text, select the passage that answers the query. If no passage "
        "answers it, choose 'None of these passages'."
    ),
    (
        "Find the passage among the options that answers the query, relying only on the text given. When none of "
        "them answers the query, choose 'None of these passages'."
    ),
    (
        "Below is a query, and the options are passages. Choose the passage that answers the query from the "
        "supplied text alone, or 'None of these passages' when none does."
    ),
]


def wikiqa_samples(row: dict) -> list[dict]:
    # Conflicting annotations of identical sentences cannot yield a reliable choice.
    passages = annotated_passages(row)
    if passages is None:
        return []
    positive, negative = passages
    seed = row["query"] + "\n" + "\n".join(sorted(positive + negative))
    # Other annotated positives are omitted, never turned into distractors.
    selected = [(text, True) for text in stable_shuffle(positive, seed)[:1]]
    selected += [(text, False) for text in stable_shuffle(negative, seed)[:MAX_DISTRACTORS]]
    if len(selected) < 2:
        return []
    selected = stable_shuffle(selected, seed)
    ids = list(ascii_uppercase[: len(selected)])
    # The passages are the options themselves, so the state holds only the query.
    options = make_options(ids, [text for text, _ in selected])
    options += make_options(["none"], ["None of these passages"])
    label = next((key for key, (_, relevant) in zip(ids, selected) if relevant), "none")
    state = f"{pick_about(ABOUT, seed)}\n\nQuery: {row['query']}"
    return [make_sample(state, pick_question(QUESTIONS, seed), stable_shuffle(options, seed), label)]


NOUL_QUESTIONS = [
    "Does the passage answer the query?",
    "Is the query answered by this passage?",
    "Does this passage contain the requested information?",
    "Can the query be answered from this passage?",
    "Does the passage provide an answer to the query?",
]
NOUL_ABOUT = [
    "Decide whether the passage answers the query, using only the supplied text.",
    "Using only the supplied text, say whether the passage answers the query.",
    "Below are a query and a passage. Judge from the passage alone whether it answers the query.",
    "Check whether the passage contains the answer to the query, relying only on the text given.",
    "Read the query and the passage, and decide from the supplied text alone whether the passage answers it.",
]
YES_NO = make_options(["yes", "no"])


def annotated_passages(row: dict) -> tuple[list[str], list[str]] | None:
    """Positive and negative passages of a query, or None if identical sentences got conflicting labels."""
    labels: dict[str, set[int]] = defaultdict(set)
    for passage in row["passages"]:
        text = passage["text"].strip()
        if text:
            labels[text].add(passage["label"])
    if not row["query"].strip() or any(len(values) != 1 for values in labels.values()):
        return None
    positive = sorted(text for text, values in labels.items() if values == {1})
    negative = sorted(text for text, values in labels.items() if values == {0})
    return positive, negative


def wikiqa_noul_samples(row: dict) -> list[dict]:
    passages = annotated_passages(row)
    if passages is None:
        return []
    positive, negative = passages
    seed = row["query"] + "\n" + "\n".join(positive + negative)
    # Most sentences do not answer the query; cap negatives so "no" does not swamp the set.
    selected = [(text, "yes") for text in positive]
    selected += [(text, "no") for text in stable_shuffle(negative, seed)[: max(2, 2 * len(positive))]]
    about = pick_about(NOUL_ABOUT, seed)
    samples = []
    for text, label in selected:
        state = f"{about}\n\nQuery: {row['query']}\nPassage: {text}"
        sample = make_sample(state, pick_question(NOUL_QUESTIONS, seed + text), YES_NO, label)
        # Each passage is its own state; the group keeps a query's rows in one split.
        samples.append({**sample, "group": row["query"]})
    return samples


def grouped_queries(data: DatasetDict) -> DatasetDict:
    """One row per query with its labelled passages. Shared Wikipedia articles and repeated queries stay out of
    lower-priority splits."""
    grouped = {}
    seen_queries: set[str] = set()
    seen_titles: set[str] = set()
    for split in ("test", "validation", "train"):
        queries = {}
        titles = defaultdict(set)
        for row in data[split]:
            if row["label"] not in (0, 1):
                continue
            key = row["question_id"]
            query = queries.setdefault(key, {"query": row["question"], "passages": []})
            query["passages"].append({"text": row["answer"], "label": row["label"]})
            titles[key].add(row["document_title"])
        kept = [
            row for key, row in queries.items() if row["query"] not in seen_queries and not titles[key] & seen_titles
        ]
        seen_queries.update(row["query"] for row in queries.values())
        seen_titles.update(title for values in titles.values() for title in values)
        grouped[split] = Dataset.from_list(kept)
    return DatasetDict({split: grouped[split] for split in data})


class WikiQADataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        self.data = grouped_queries(self.data)
        self.data = self.data.map(
            explode(wikiqa_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


class WikiQANoulDataset(JevDataset):
    type = DatasetType.NOUL

    def prepare(self):
        self.data = grouped_queries(self.data)
        self.data = self.data.map(
            explode(wikiqa_noul_samples, GROUPED_FEATURES),
            batched=True,
            remove_columns=self.source_columns,
            features=GROUPED_FEATURES,
        )


wikiqa_dataset = WikiQADataset(name="wikiqa", hf_path="microsoft/wiki_qa")
wikiqa_noul_dataset = WikiQANoulDataset(name="wikiqa_noul", hf_path="microsoft/wiki_qa")
