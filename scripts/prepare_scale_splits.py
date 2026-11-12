"""Create deterministic train-size subsets from the validated augmented set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/splits/train_augmented.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/splits/scales_balanced"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[7, 14, 28])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = read_jsonl(args.input)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.task_type].append(record)
    task_types = sorted(grouped)
    shuffled = {}
    for task_type in task_types:
        values = list(grouped[task_type])
        random.Random(f"{args.seed}:{task_type}").shuffle(values)
        shuffled[task_type] = values

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        if size > len(records):
            raise ValueError(f"requested {size} records but input has {len(records)}")
        exact = {task_type: size * len(grouped[task_type]) / len(records) for task_type in task_types}
        counts = {task_type: int(value) for task_type, value in exact.items()}
        remaining = size - sum(counts.values())
        ranked = sorted(task_types, key=lambda task: (exact[task] - counts[task], task), reverse=True)
        for task_type in ranked[:remaining]:
            counts[task_type] += 1
        selected = [record for task_type in task_types for record in shuffled[task_type][:counts[task_type]]]
        random.Random(f"{args.seed}:{size}:order").shuffle(selected)
        output = output_dir / f"train_{size}.jsonl"
        write_jsonl(output, selected)
        manifest = {
            "input": str(Path(args.input)),
            "seed": args.seed,
            "requested_size": size,
            "counts_by_task_type": counts,
            "example_ids": [record.example_id for record in selected],
        }
        output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote={output} records={len(selected)} task_types={counts}")


if __name__ == "__main__":
    main()
