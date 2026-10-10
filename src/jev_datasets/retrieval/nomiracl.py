import csv
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, DownloadManager
from loguru import logger

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, stable_shuffle

# Pin all nine files together so a Hub update cannot mix corpus and judgments.
SOURCE_REVISION = "ecd08778d0426a5ca28ac99763b0c9ddc2c78e68"
PASSAGE_COUNT = 10
# The texts explicitly name the USSR, but this query is marked non-relevant.
# Quarantine rather than relabel a reviewed source error (docs/dataset-reviews/nomiracl.md).
EXCLUDED_QUERIES = {("russian", "validation", "335882#0")}

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
    passages = stable_shuffle(sorted(row["passages"], key=lambda p: (p["title"], p["text"])), seed)
    # All judged passages remain in the input: subsampling could remove the only answer.
    text = "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))
    if language == "russian":
        about = (
            "Определите, содержат ли приведённые фрагменты ответ на запрос. Используйте только эти тексты. "
            "«Да» означает, что запрошенная информация есть, даже если сам ответ на запрос отрицательный. "
            "«Нет» означает, что информации для ответа недостаточно; одного совпадения темы недостаточно."
        )
        state = f"{about}\n\nЗапрос: {row['query']}\n\nФрагменты:\n{text}"
        options = make_options(["yes", "no"], ["да", "нет"])
    else:
        about = (
            "Determine whether the supplied passages answer the query. Use only these texts. "
            "Yes means the requested information is present, even if the query's answer is negative. "
            "No means there is not enough information to answer; merely sharing the topic is insufficient."
        )
        state = f"{about}\n\nQuery: {row['query']}\n\nPassages:\n{text}"
        options = make_options(["yes", "no"])
    label = "yes" if any(p["relevance"] == 1 for p in passages) else "no"
    return make_sample(state, pick_question(QUESTIONS[language], seed), options, label)


class NoMIRACLDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        root = f"https://huggingface.co/datasets/{self.hf_path}/resolve/{SOURCE_REVISION}/data/{self.hf_name}"
        files = {"corpus": f"{root}/corpus.jsonl.gz"}
        for split in ("dev", "test"):
            for subset in ("relevant", "non_relevant"):
                for kind in ("topics", "qrels"):
                    files[f"{split}.{subset}.{kind}"] = f"{root}/{kind}/{split}.{subset}.tsv"
        # Read the official compact corpus directly; datasets 4+ cannot execute its legacy loader.
        paths = DownloadManager().download_and_extract(files)
        corpus = {}
        with Path(paths["corpus"]).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                docid = row["docid"]
                if docid in corpus and corpus[docid] != row:
                    raise ValueError(f"Conflicting NoMIRACL corpus entries for {docid}")
                corpus[docid] = row
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
                with Path(paths[f"{prefix}.qrels"]).open(encoding="utf-8") as handle:
                    for qid, _, docid, relevance in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                        if relevance not in ("0", "1"):
                            raise ValueError(f"Unexpected relevance: {relevance}")
                        if docid in judgments[qid] and judgments[qid][docid] != int(relevance):
                            raise ValueError(f"Conflicting NoMIRACL judgments for {prefix}/{qid}/{docid}")
                        judgments[qid][docid] = int(relevance)
                with Path(paths[f"{prefix}.topics"]).open(encoding="utf-8") as handle:
                    for qid, query in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                        labels = judgments[qid]
                        if missing := labels.keys() - corpus.keys():
                            raise ValueError(f"Missing NoMIRACL evidence for {prefix}/{qid}: {sorted(missing)}")
                        # Missing/blank evidence cannot silently turn an answerable query into a negative.
                        if not query.strip() or not labels or any(not corpus[d]["text"].strip() for d in labels):
                            continue
                        if any(labels.values()) != (subset == "relevant"):
                            raise ValueError(f"Inconsistent NoMIRACL subset for {prefix}/{qid}")
                        if query in seen_queries or labels.keys() & seen_docs:
                            continue
                        passages = [dict(corpus[docid], relevance=label) for docid, label in labels.items()]
                        rows.append({"query_id": qid, "query": query, "passages": passages})
                        split_queries.add(query)
                        split_docs.update(labels)
            seen_queries.update(split_queries)
            seen_docs.update(split_docs)
            data[split] = Dataset.from_list(rows)
        # NoMIRACL is an evaluation benchmark. Never expose its dev/test rows as training data.
        return DatasetDict(data)

    def prepare(self):
        # Dev qrels can contain 11–18 passages, exclusively for positives. Keep only complete
        # ten-passage inputs to remove this label cue without selecting evidence using gold.
        counts = self.data.num_rows
        self.data = DatasetDict(
            {
                split: rows.filter(
                    lambda row: (
                        len(row["passages"]) == PASSAGE_COUNT
                        and (self.hf_name, split, row["query_id"]) not in EXCLUDED_QUERIES
                    )
                )
                for split, rows in self.data.items()
            }
        )
        unique = {}
        for split, rows in self.data.items():
            groups = defaultdict(list)
            targets = defaultdict(set)
            for i, row in enumerate(rows):
                # Ignore qrel/docid ordering: the same visible evidence cannot have opposing labels.
                key = (row["query"], tuple(sorted((p["title"], p["text"]) for p in row["passages"])))
                groups[key].append(i)
                targets[key].add(any(p["relevance"] == 1 for p in row["passages"]))
            conflicts = sum(len(indices) for key, indices in groups.items() if len(targets[key]) > 1)
            duplicates = sum(len(indices) - 1 for key, indices in groups.items() if len(targets[key]) == 1)
            unique[split] = rows.select(sorted(indices[0] for key, indices in groups.items() if len(targets[key]) == 1))
            logger.info("{}/{}: removed {} conflicting rows and {} duplicates", self.name, split, conflicts, duplicates)
        self.data = DatasetDict(unique)
        logger.info("{}: complete {}-passage rows {} -> {}", self.name, PASSAGE_COUNT, counts, self.data.num_rows)
        self.data = self.data.map(
            lambda row: nomiracl_sample(row, self.hf_name),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


class EnglishNoMIRACLDataset(NoMIRACLDataset):
    type = DatasetType.NOUL


nomiracl_en_dataset = EnglishNoMIRACLDataset(name="nomiracl_en", hf_path="miracl/nomiracl", hf_name="english")
nomiracl_ru_dataset = NoMIRACLDataset(name="nomiracl_ru", hf_path="miracl/nomiracl", hf_name="russian")
