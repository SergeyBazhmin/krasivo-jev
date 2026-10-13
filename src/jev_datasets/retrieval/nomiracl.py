import csv
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import GROUPED_FEATURES, explode, make_options, make_sample, pick_question, stable_shuffle

QUESTIONS = {
    "english": [
        "Does any supplied passage answer the query?",
        "Can the query be answered using these passages?",
        "Is an answer to the query present in the supplied text?",
        "Does at least one passage provide the requested information?",
        "Do these passages contain an answer to the query?",
    ],
    "russian": [
        "Есть ли ответ на запрос хотя бы в одном из приведённых фрагментов?",
        "Можно ли ответить на запрос по этим фрагментам?",
        "Содержится ли ответ на запрос в приведённых текстах?",
        "Даёт ли хотя бы один фрагмент запрошенную информацию?",
        "Есть ли среди этих фрагментов текст с ответом на запрос?",
    ],
}
PASSAGE_QUESTIONS = {
    "english": [
        "Does this passage answer the query?",
        "Can the query be answered using this passage?",
        "Is an answer to the query present in the passage?",
        "Does the passage provide the requested information?",
        "Does this text contain an answer to the query?",
    ],
    "russian": [
        "Отвечает ли этот фрагмент на запрос?",
        "Можно ли ответить на запрос по этому фрагменту?",
        "Содержится ли ответ на запрос в приведённом фрагменте?",
        "Даёт ли фрагмент запрошенную информацию?",
        "Есть ли в этом тексте ответ на запрос?",
    ],
}
CHOICE_QUESTIONS = {
    "english": [
        "Which passage answers the query?",
        "Select the passage that provides an answer to the query.",
        "Which of these passages contains the requested information?",
        "Choose the passage that answers the query.",
        "In which passage can the query be answered?",
    ],
    "russian": [
        "Какой фрагмент отвечает на запрос?",
        "Выберите фрагмент, в котором есть ответ на запрос.",
        "В каком из этих фрагментов содержится запрошенная информация?",
        "Укажите фрагмент, который отвечает на запрос.",
        "По какому фрагменту можно ответить на запрос?",
    ],
}
CHOICE_ABOUT = {
    "english": (
        "Each option is a passage. Select the passage that answers the query, using only the supplied text. "
        "Choose 'None of these passages' if none answers the query."
    ),
    "russian": (
        "Каждый вариант ответа — фрагмент текста. Выберите фрагмент, который отвечает на запрос, используя только "
        "приведённые тексты. Если ни один фрагмент не отвечает на запрос, выберите «Ни один из фрагментов»."
    ),
}
NONE_TEXT = {"english": "None of these passages", "russian": "Ни один из фрагментов"}


def yes_no(language: str) -> list[dict[str, str]]:
    return make_options(["yes", "no"], ["да", "нет"] if language == "russian" else None)


def nomiracl_sample(row: dict, language: str) -> dict:
    seed = row["query"] + "\n" + "\n".join(sorted(p["text"] for p in row["passages"]))
    passages = stable_shuffle(row["passages"], seed)
    # All judged passages remain in the input: subsampling could remove the only answer.
    text = "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))
    if language == "russian":
        about = "Определите, содержат ли приведённые фрагменты ответ на запрос. Используйте только эти тексты."
        state = f"{about}\n\nЗапрос: {row['query']}\n\nФрагменты:\n{text}"
    else:
        about = "Determine whether the supplied passages answer the query. Use only these texts."
        state = f"{about}\n\nQuery: {row['query']}\n\nPassages:\n{text}"
    label = "yes" if any(p["relevance"] == 1 for p in passages) else "no"
    return make_sample(state, pick_question(QUESTIONS[language], seed), yes_no(language), label)


def nomiracl_passage_samples(row: dict, language: str) -> list[dict]:
    samples = []
    for passage in row["passages"]:
        seed = row["query"] + "\n" + passage["text"]
        text = f"{passage['title']}\n{passage['text']}"
        if language == "russian":
            about = "Определите, содержит ли приведённый фрагмент ответ на запрос. Используйте только этот текст."
            state = f"{about}\n\nЗапрос: {row['query']}\n\nФрагмент:\n{text}"
        else:
            about = "Determine whether the supplied passage answers the query. Use only this text."
            state = f"{about}\n\nQuery: {row['query']}\n\nPassage:\n{text}"
        label = "yes" if passage["relevance"] == 1 else "no"
        samples.append(make_sample(state, pick_question(PASSAGE_QUESTIONS[language], seed), yes_no(language), label))
    return samples


def nomiracl_choice_samples(row: dict, language: str) -> list[dict]:
    # Identical passage texts with conflicting judgments cannot yield a reliable choice.
    labels: dict[str, set[int]] = defaultdict(set)
    for passage in row["passages"]:
        labels[f"{passage['title']}\n{passage['text']}"].add(passage["relevance"])
    if any(len(values) != 1 for values in labels.values()):
        return []
    positive = sorted(text for text, values in labels.items() if values == {1})
    negative = sorted(text for text, values in labels.items() if values == {0})
    seed = row["query"] + "\n" + "\n".join(sorted(labels))
    # Other relevant passages are omitted, never turned into distractors.
    selected = [(text, True) for text in stable_shuffle(positive, seed)[:1]]
    selected += [(text, False) for text in stable_shuffle(negative, seed)[: 4 - len(selected)]]
    if len(selected) < 2:
        return []
    selected = stable_shuffle(selected, seed)
    ids = list("ABCD")[: len(selected)]
    options = make_options(ids, [text for text, _ in selected])
    options += make_options(["none"], [NONE_TEXT[language]])
    label = next((key for key, (_, relevant) in zip(ids, selected) if relevant), "none")
    query = "Запрос" if language == "russian" else "Query"
    # The passages are the options themselves, so the state holds only the query.
    state = f"{CHOICE_ABOUT[language]}\n\n{query}: {row['query']}"
    return [make_sample(state, pick_question(CHOICE_QUESTIONS[language], seed), stable_shuffle(options, seed), label)]


class NoMIRACLDataset(JevDataset):
    type = DatasetType.NOUL

    def load(self) -> DatasetDict:
        root = f"https://huggingface.co/datasets/{self.hf_path}/resolve/main/data/{self.hf_name}"
        files = {"corpus": f"{root}/corpus.jsonl.gz"}
        for split in ("dev", "test"):
            for subset in ("relevant", "non_relevant"):
                for kind in ("topics", "qrels"):
                    files[f"{split}.{subset}.{kind}"] = f"{root}/{kind}/{split}.{subset}.tsv"
        # Read the official compact corpus directly; datasets 4+ cannot execute its legacy loader.
        paths = DownloadManager().download_and_extract(files)
        with Path(paths["corpus"]).open() as handle:
            corpus = {row["docid"]: row for line in handle if (row := json.loads(line))}
        data = {}
        seen_queries: set[str] = set()
        seen_docs: set[str] = set()
        for source_split, split in (("test", "test"), ("dev", "validation")):
            rows = []
            split_docs: set[str] = set()
            split_queries: set[str] = set()
            for subset in ("relevant", "non_relevant"):
                prefix = f"{source_split}.{subset}"
                judgments = defaultdict(dict)
                with Path(paths[f"{prefix}.qrels"]).open() as handle:
                    for qid, _, docid, relevance in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                        if relevance not in ("0", "1"):
                            raise ValueError(f"Unexpected relevance: {relevance}")
                        judgments[qid][docid] = int(relevance)
                with Path(paths[f"{prefix}.topics"]).open() as handle:
                    for qid, query in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                        labels = judgments[qid]
                        # Missing/blank evidence cannot silently turn an answerable query into a negative.
                        if not query.strip() or not labels or any(not corpus[d]["text"].strip() for d in labels):
                            continue
                        if any(labels.values()) != (subset == "relevant"):
                            raise ValueError(f"Inconsistent NoMIRACL subset for {prefix}/{qid}")
                        if query in seen_queries or labels.keys() & seen_docs:
                            continue
                        passages = [dict(corpus[docid], relevance=label) for docid, label in labels.items()]
                        rows.append({"query": query, "passages": passages})
                        split_queries.add(query)
                        split_docs.update(labels)
            seen_queries.update(split_queries)
            seen_docs.update(split_docs)
            data[split] = Dataset.from_list(rows)
        # NoMIRACL is an evaluation benchmark. Never expose its dev/test rows as training data.
        return DatasetDict(data)

    def samples(self, row: dict) -> list[dict]:
        return [nomiracl_sample(row, self.hf_name)]

    def prepare(self):
        # Every variant groups by query, so a query lands in the same split in all of them.
        self.data = self.data.map(
            explode(lambda row: [{**s, "group": row["query"]} for s in self.samples(row)], GROUPED_FEATURES),
            batched=True,
            remove_columns=self.source_columns,
            features=GROUPED_FEATURES,
        )


class NoMIRACLPassageDataset(NoMIRACLDataset):
    def samples(self, row: dict) -> list[dict]:
        return nomiracl_passage_samples(row, self.hf_name)


class NoMIRACLChoiceDataset(NoMIRACLDataset):
    type = DatasetType.CHOICE

    def samples(self, row: dict) -> list[dict]:
        return nomiracl_choice_samples(row, self.hf_name)


nomiracl_en_dataset = NoMIRACLDataset(name="nomiracl_en", hf_path="miracl/nomiracl", hf_name="english")
nomiracl_ru_dataset = NoMIRACLDataset(name="nomiracl_ru", hf_path="miracl/nomiracl", hf_name="russian")
nomiracl_passage_en_dataset = NoMIRACLPassageDataset(
    name="nomiracl_passage_en", hf_path="miracl/nomiracl", hf_name="english"
)
nomiracl_passage_ru_dataset = NoMIRACLPassageDataset(
    name="nomiracl_passage_ru", hf_path="miracl/nomiracl", hf_name="russian"
)
nomiracl_choice_en_dataset = NoMIRACLChoiceDataset(
    name="nomiracl_choice_en", hf_path="miracl/nomiracl", hf_name="english"
)
nomiracl_choice_ru_dataset = NoMIRACLChoiceDataset(
    name="nomiracl_choice_ru", hf_path="miracl/nomiracl", hf_name="russian"
)
