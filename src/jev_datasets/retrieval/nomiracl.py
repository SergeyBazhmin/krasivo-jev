import csv
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, stable_shuffle

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


def nomiracl_sample(row: dict, language: str) -> dict:
    seed = row["query"] + "\n" + "\n".join(sorted(p["text"] for p in row["passages"]))
    passages = stable_shuffle(row["passages"], seed)
    # All judged passages remain in the input: subsampling could remove the only answer.
    text = "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))
    if language == "russian":
        about = "Определите, содержат ли приведённые фрагменты ответ на запрос. Используйте только эти тексты."
        state = f"{about}\n\nЗапрос: {row['query']}\n\nФрагменты:\n{text}"
        options = make_options(["yes", "no"], ["да", "нет"])
    else:
        about = "Determine whether the supplied passages answer the query. Use only these texts."
        state = f"{about}\n\nQuery: {row['query']}\n\nPassages:\n{text}"
        options = make_options(["yes", "no"])
    label = "yes" if any(p["relevance"] == 1 for p in passages) else "no"
    return make_sample(state, pick_question(QUESTIONS[language], seed), options, label)


class NoMIRACLDataset(JevDataset):
    type = DatasetType.CHOICE

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

    def prepare(self):
        self.data = self.data.map(
            lambda row: nomiracl_sample(row, self.hf_name),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


class EnglishNoMIRACLDataset(NoMIRACLDataset):
    type = DatasetType.NOUL


nomiracl_en_dataset = EnglishNoMIRACLDataset(name="nomiracl_en", hf_path="miracl/nomiracl", hf_name="english")
nomiracl_ru_dataset = NoMIRACLDataset(name="nomiracl_ru", hf_path="miracl/nomiracl", hf_name="russian")
