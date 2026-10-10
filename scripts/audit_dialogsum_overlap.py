"""Candidate search using five-word shingles; verify with ordered token similarity."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from jev_datasets.structured.dialogsum import REVISION
from jev_datasets.structured.source import content_group


def words(text: str) -> list[str]:
    return re.findall(r"\w+", re.sub(r"#Person\d+#:", "", text).casefold())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    metadata = {}
    for split in ("train", "validation", "test"):
        path = getattr(args, split)
        sources[split] = [json.loads(line) for line in path.read_text().splitlines()]
        metadata[split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(sources[split])}
    docs = []
    inverted = defaultdict(list)
    for split, rows in sources.items():
        for i, row in enumerate(rows):
            text = row["dialogue"]
            tokens = words(text)
            shingles = {tuple(tokens[j : j + 5]) for j in range(len(tokens) - 4)}
            doc = {
                "locator": f"{split}/{i}/{row['fname']}",
                "split": split,
                "group": content_group(text.strip()),
                "tokens": tokens,
                "shingles": shingles,
                "text": text,
            }
            n = len(docs)
            docs.append(doc)
            for shingle in shingles:
                inverted[shingle].append(n)
    pairs = []
    for n, doc in enumerate(docs):
        if doc["split"] == "train":
            continue
        candidates = Counter(j for shingle in doc["shingles"] for j in inverted[shingle] if j != n)
        for j, common in candidates.items():
            peer = docs[j]
            if peer["split"] == doc["split"] or (peer["split"] != "train" and j < n):
                continue
            containment = common / min(len(doc["shingles"]), len(peer["shingles"]))
            if common < 10 or containment < 0.5:
                continue
            ratio = SequenceMatcher(None, doc["tokens"], peer["tokens"], autojunk=False).ratio()
            short, long = sorted((doc["tokens"], peer["tokens"]), key=len)
            contained = len(short) >= 40 and f" {' '.join(short)} " in f" {' '.join(long)} "
            if ratio < 0.85 and not contained:
                continue
            pairs.append(
                {
                    "left": {k: doc[k] for k in ("locator", "split", "group", "text")},
                    "right": {k: peer[k] for k in ("locator", "split", "group", "text")},
                    "ratio": ratio,
                    "five_word_containment": containment,
                    "whole_shorter_transcript_contained": contained,
                }
            )
    priority = {"test": 0, "validation": 1, "train": 2}
    excluded = {}
    for pair in pairs:
        discard = max((pair["left"], pair["right"]), key=lambda doc: priority[doc["split"]])
        excluded[discard["group"]] = discard["split"]
    manifest = {
        "revision": REVISION,
        "source_files": metadata,
        "method": "Ignore speaker tags, case and punctuation; candidates share >=10 five-word shingles and >=50% of the smaller shingle set. Exclude lower-priority transcript when ordered token SequenceMatcher(autojunk=False) ratio >=0.85, or a >=40-token transcript is contained verbatim. Priority: test > validation > train. Hash: content_group(dialogue.strip()).",
        "pair_counts": dict(Counter(":".join(sorted((p["left"]["split"], p["right"]["split"]))) for p in pairs)),
        "excluded_dialogues": dict(sorted(excluded.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    args.details.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "split_pairs": dict(
                    Counter(":".join(sorted((p["left"]["split"], p["right"]["split"]))) for p in pairs)
                ),
                "ratio_bins": dict(
                    Counter(
                        "exact_words"
                        if p["ratio"] == 1
                        else ">=.95"
                        if p["ratio"] >= 0.95
                        else ">=.9"
                        if p["ratio"] >= 0.9
                        else ">=.85"
                        if p["ratio"] >= 0.85
                        else "contained_<.85"
                        for p in pairs
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
