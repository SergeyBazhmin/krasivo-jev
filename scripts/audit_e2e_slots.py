"""Reproduce the E2E review using pinned CSVs and already built datasets; no downloads or inference."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, disable_progress_bars, load_from_disk

from jev_datasets.structured.e2e import (
    FIELDS,
    REVISION,
    affirmed_literal,
    clean_records,
    context_key,
    e2e_samples,
    literal,
    parse_mr,
    slot_literal,
)
from jev_datasets.structured.source import content_group, record


def gold(sample: dict) -> str:
    return sample["options"][sample["label"].index(1.0)]["text"]


def fixtures() -> dict:
    pools = {
        "name": ["The Eagle", "The Mill", "Café Sicilia", "The Bakers"],
        "near": ["The Eagle", "The Mill", "Café Sicilia", "The Bakers"],
        "food": ["Italian", "Chinese", "French"],
        "eatType": ["pub", "coffee shop", "restaurant"],
        "area": ["city centre", "riverside"],
    }
    assert parse_mr("name[A], name[B]") is None
    assert parse_mr("name[A] garbage") is None
    assert not literal("pub", "public garden")
    assert not affirmed_literal("Italian", "The Eagle does not serve Italian food.")
    assert not affirmed_literal("Chinese", "It doesn't serve Chinese food.")
    assert affirmed_literal("Italian", "Not only Italian food but also good wine.")
    assert affirmed_literal("pub", "The Eagle is not a family-friendly pub.")
    assert not slot_literal("food", "English", "It is an English speaking establishment.")
    assert not slot_literal("food", "French", "The staff speaks French.")
    assert slot_literal("food", "English", "It is English speaking and serves English food.")
    text = "The Eagle is a pub near Café Sicilia serving Italian food downtown."
    fields = {"name": "The Eagle", "near": "Café Sicilia", "food": "Italian", "area": "city centre"}
    row = record({"fields": fields, "text": text}, "fixture")
    samples = e2e_samples(row, pools)
    assert samples == e2e_samples(row, pools)
    assert len(samples) == 3  # The unmentioned area must not consume the three-question budget.
    for sample in samples:
        if gold(sample) in {"The Eagle", "Café Sicilia"}:
            assert {"The Eagle", "Café Sicilia"} <= {o["text"] for o in sample["options"]}
    named = record({"fields": {"name": "The French House", "food": "French"}, "text": "The French House."}, "n")
    assert all(gold(s) != "French" for s in e2e_samples(named, pools))
    negated = record(
        {"fields": {"name": "The Eagle", "food": "Italian"}, "text": "The Eagle does not serve Italian food."}, "n"
    )
    assert all(gold(s) != "Italian" for s in e2e_samples(negated, pools))

    def raw(text: str) -> dict:
        return record({"fields": {"name": "The Eagle"}, "text": text}, "n")

    data = DatasetDict(
        train=Dataset.from_list([raw("THE EAGLE!"), raw("The Mill")]),
        validation=Dataset.from_list([raw("The Eagle.")]),
        test=Dataset.from_list([raw("The Eagle"), raw("the eagle!")]),
    )
    cleaned = clean_records(data)
    assert {k: len(v) for k, v in cleaned.items()} == {"train": 1, "validation": 0, "test": 1}
    assert {k: v.to_dict() for k, v in clean_records(cleaned).items()} == {k: v.to_dict() for k, v in cleaned.items()}
    return {
        "passed": True,
        "checks": "parsing, boundaries, negation, role negatives, determinism, budget, deduplication",
    }


def build_metrics(path: Path) -> dict:
    data = load_from_disk(path)
    result = {}
    for split, rows in data.items():
        counts = Counter()
        present_choice_accuracy = 0.0
        for sample in rows:
            field = next(k for k, value in FIELDS.items() if value in sample["question"])
            counts[field] += 1
            description = sample["state"].split("\n\n", 1)[1]
            present = [o["text"] for o in sample["options"] if literal(o["text"], description)]
            assert gold(sample) in present
            if len(present) > 1:
                counts["multiple_present"] += 1
                counts[f"multiple_present_{field}"] += 1
            present_choice_accuracy += 1 / len(present)
        result[split] = {
            "rows": len(rows),
            "counts": dict(counts),
            "presence_only_expected_accuracy": present_choice_accuracy / len(rows),
            "uniform_expected_accuracy": sum(1 / len(s["options"]) for s in rows) / len(rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="source-version.json with local pinned CSV paths")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    disable_progress_bars()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["revision"] == REVISION
    report = {"revision": REVISION, "fixtures": fixtures(), "source": {}, "builds": {}}
    groups = defaultdict(lambda: defaultdict(set))
    anomalies = []
    wrong_roles = []
    for split, path in manifest["paths"].items():
        file = Path(path)
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest == manifest["source"][split]["sha256"]
        with file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        counts = Counter()
        for index, row in enumerate(rows):
            fields = parse_mr(row["mr"])
            if fields is None:
                counts["invalid_or_multivalued_mr"] += 1
                continue
            groups["mr"][split].add(content_group(fields))
            groups["context"][split].add(context_key(row["ref"]))
            text = row["ref"]
            for entity in (fields.get("name", ""), fields.get("near", "")):
                if entity:
                    text = re.sub(re.escape(entity), "ENTITY", text, flags=re.IGNORECASE)
            groups["delexicalized_context"][split].add(context_key(text))
            for field in fields.keys() & FIELDS.keys():
                evidence = row["ref"]
                if field not in {"name", "near"}:
                    for entity in (fields.get("name", ""), fields.get("near", "")):
                        if entity:
                            evidence = re.sub(re.escape(entity), "", evidence, flags=re.IGNORECASE)
                value = fields[field]
                if literal(value, evidence) and not affirmed_literal(value, evidence):
                    anomalies.append({"split": split, "raw_index": index, "field": field, "value": value, "row": row})
                if affirmed_literal(value, evidence) and not slot_literal(field, value, evidence):
                    wrong_roles.append({"split": split, "raw_index": index, "field": field, "value": value, "row": row})
            for field in ("name", "near"):
                if field in fields:
                    groups[field][split].add(fields[field].casefold())
        report["source"][split] = {"rows": len(rows), "sha256": digest, "filters": dict(counts)}
    report["overlaps"] = {
        group: {
            f"{a}:{b}": len(sets[a] & sets[b])
            for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
        }
        for group, sets in groups.items()
    }
    report["direct_negation_anomalies"] = anomalies
    report["wrong_role_anomalies"] = wrong_roles
    report["anomaly_conversion"] = {}
    for name, path in [("before", args.before), ("after", args.after)]:
        if path is None:
            continue
        built = load_from_disk(path)
        bad_inputs = {(a["split"], a["row"]["ref"], a["field"]): a for a in anomalies + wrong_roles}
        retained = []
        for split, rows in built.items():
            for sample in rows:
                description = sample["state"].split("\n\n", 1)[1]
                field = next(k for k, value in FIELDS.items() if value in sample["question"])
                if (split, description, field) in bad_inputs:
                    retained.append({"split": split, "description": description, "field": field})
        report["anomaly_conversion"][name] = retained
        if name == "after":
            assert not retained
    if args.before:
        report["builds"]["before"] = build_metrics(args.before)
    report["builds"]["after"] = build_metrics(args.after)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
