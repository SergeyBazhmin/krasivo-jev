from collections import defaultdict

from datasets import Dataset, DatasetDict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample, pick_question, stable_shuffle

QUESTIONS = [
    "Which passage answers the query?",
    "Select the passage that provides an answer to the query.",
    "Which of these passages contains the requested information?",
    "Choose the passage that answers the supplied question.",
    "In which passage can the query be answered?",
]
ABOUT = (
    "Select an answer-bearing passage using only the supplied text. "
    "Choose 'None of these passages' if none answers the query."
)


def wikiqa_samples(row: dict) -> list[dict]:
    # Conflicting annotations of identical sentences cannot yield a reliable choice.
    labels: dict[str, set[int]] = defaultdict(set)
    for passage in row["passages"]:
        text = passage["text"].strip()
        if text:
            labels[text].add(passage["label"])
    if not row["query"].strip() or any(len(values) != 1 for values in labels.values()):
        return []
    positive = sorted(text for text, values in labels.items() if values == {1})
    negative = sorted(text for text, values in labels.items() if values == {0})
    seed = row["query"] + "\n" + "\n".join(sorted(labels))
    # Other annotated positives are omitted, never turned into distractors.
    selected = [(text, True) for text in stable_shuffle(positive, seed)[:1]]
    selected += [(text, False) for text in stable_shuffle(negative, seed)[: 4 - len(selected)]]
    if len(selected) < 2:
        return []
    selected = stable_shuffle(selected, seed)
    ids = list("ABCD")[: len(selected)]
    passages = "\n\n".join(f"Passage {key}:\n{text}" for key, (text, _) in zip(ids, selected))
    # Assign positional ids after shuffling, so the gold id is not always A.
    options = make_options(ids, [f"Passage {key}" for key in ids])
    options += make_options(["none"], ["None of these passages"])
    label = next((key for key, (_, relevant) in zip(ids, selected) if relevant), "none")
    state = f"{ABOUT}\n\nQuery: {row['query']}\n\n{passages}"
    return [make_sample(state, pick_question(QUESTIONS, seed), stable_shuffle(options, seed), label)]


class WikiQADataset(JevDataset):
    type = DatasetType.CHOICE

    def prepare(self):
        data = self.data
        grouped = {}
        seen_queries: set[str] = set()
        seen_titles: set[str] = set()
        # Shared Wikipedia articles and repeated queries stay out of lower-priority splits.
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
                row
                for key, row in queries.items()
                if row["query"] not in seen_queries and not titles[key] & seen_titles
            ]
            seen_queries.update(row["query"] for row in queries.values())
            seen_titles.update(title for values in titles.values() for title in values)
            grouped[split] = Dataset.from_list(kept)
        self.data = DatasetDict({split: grouped[split] for split in data})
        self.data = self.data.map(
            explode(wikiqa_samples), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


wikiqa_dataset = WikiQADataset(name="wikiqa", hf_path="microsoft/wiki_qa")
