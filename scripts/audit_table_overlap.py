"""Write the exact WikiSQL/ToTTo table-overlap manifest from their official archives.

Run: .venv/bin/python scripts/audit_table_overlap.py --wikisql data.tar.bz2 --totto totto_data.zip --output manifest.json
"""

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

from jev_datasets.structured.wikisql import REVISION, normalized_table, table_group


def archive_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def overlaps(wikisql: Path, totto: Path) -> dict:
    wiki_groups = set()
    with tarfile.open(wikisql) as archive:
        for split in ("train", "dev", "test"):
            for line in archive.extractfile(f"data/{split}.tables.jsonl"):
                if table := normalized_table(json.loads(line)):
                    wiki_groups.add(table_group(table))
    shared = set()
    with zipfile.ZipFile(totto) as archive:
        for name in archive.namelist():
            if not name.endswith(".jsonl"):
                continue
            for line in archive.open(name):
                cells = json.loads(line)["table"]
                if not cells or not all(cell["is_header"] for cell in cells[0]):
                    continue
                if any(cell["column_span"] != 1 or cell["row_span"] != 1 for row in cells for cell in row):
                    continue
                table = normalized_table(
                    {
                        "header": [cell["value"] for cell in cells[0]],
                        "rows": [[cell["value"] for cell in row] for row in cells[1:]],
                        "types": ["text"] * len(cells[0]),
                    }
                )
                if table is not None and (group := table_group(table)) in wiki_groups:
                    shared.add(group)
    return {
        "wikisql_revision": REVISION,
        "wikisql_sha256": archive_hash(wikisql),
        "totto_url": "https://storage.googleapis.com/totto-public/totto_data.zip",
        "totto_sha256": archive_hash(totto),
        "fingerprint": "SHA256 of casefolded headers and sorted casefolded rows; stripped cell whitespace",
        "excluded_groups": sorted(shared),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikisql", type=Path, required=True)
    parser.add_argument("--totto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = overlaps(args.wikisql, args.totto)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Excluded {len(manifest['excluded_groups'])} shared table groups; saved {args.output}")
