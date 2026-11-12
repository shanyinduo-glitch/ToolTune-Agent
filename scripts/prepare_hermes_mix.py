"""Select a deterministic, diverse Hermes review subset and build mix manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50)
    args = parser.parse_args()
    source = ROOT / "data/processed/hermes_literal_review.jsonl"
    rows = read_jsonl(source)
    groups = {}
    for row in rows:
        key = (row.metadata.get("category") or "unknown", row.metadata.get("subcategory") or "unknown")
        groups.setdefault(key, []).append(row)
    selected = []
    remaining = list(groups)
    cursor = 0
    while len(selected) < min(args.size, len(rows)) and remaining:
        key = remaining[cursor % len(remaining)]
        bucket = groups[key]
        selected.append(bucket.pop(0))
        if not bucket:
            remaining.remove(key)
            cursor = 0
        else:
            cursor += 1
    for index, row in enumerate(selected):
        row.example_id = f"hermes_review_{index:03d}"
        row.metadata.update({"split": "train_candidate", "selection": "round_robin_category_subcategory", "selection_size": len(selected)})
    out_dir = ROOT / "data/processed"
    subset = out_dir / "hermes_train_candidate_50.jsonl"
    write_jsonl(subset, selected)
    local = read_jsonl(ROOT / "data/splits/train_augmented.jsonl")
    write_jsonl(out_dir / "train_local_28.jsonl", local)
    write_jsonl(out_dir / "train_hermes_50.jsonl", selected)
    write_jsonl(out_dir / "train_local_plus_hermes_78.jsonl", local + selected)
    report = {"source_records": len(rows), "selected_records": len(selected), "unique_source_ids": len({r.metadata.get("source_id") for r in selected}), "categories": Counter(r.metadata.get("category") for r in selected), "subset": str(subset), "training_candidates": {"local": 28, "hermes": len(selected), "local_plus_hermes": 28 + len(selected)}, "approval": "review_required_before_production_training"}
    (out_dir / "hermes_train_candidate_50.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
