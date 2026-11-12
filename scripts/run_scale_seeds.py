"""Run fixed-split, multiple-training-seed scale comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/cache"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[7, 14, 28])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 123])
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()
    python = sys.executable
    metrics_dir = ROOT / "outputs/metrics"
    for seed in args.seeds:
        for size in args.sizes:
            data = ROOT / f"data/splits/scales_balanced/train_{size}.jsonl"
            if not data.exists():
                raise FileNotFoundError(f"fixed split missing: {data}; prepare seed-42 splits first")
            adapter = ROOT / f"outputs/adapters/scale-balanced-{size}-trainseed-{seed}"
            run([python, str(ROOT / "scripts/train_sft.py"), "--model", args.model, "--data", str(data), "--output", str(adapter), "--max-steps", str(args.max_steps), "--seed", str(seed)])
            for suite in ("heldout_orders", "heldout_unseen"):
                suffix = "orders" if suite == "heldout_orders" else "unseen"
                output = metrics_dir / f"scale_balanced_{size}_trainseed_{seed}_{suffix}.json"
                run([python, str(ROOT / "scripts/evaluate_baseline.py"), "--model", args.model, "--cache-dir", args.cache_dir, "--adapter", str(adapter), "--suite", suite, "--seed", str(seed), "--output", str(output)])
    summary: dict[str, object] = {"fixed_split_seed": 42, "training_seeds": args.seeds, "sizes": {}}
    for size in args.sizes:
        row = {}
        for suite in ("orders", "unseen"):
            values = []
            for seed in sorted(set(args.seeds + [42])):
                path = metrics_dir / f"scale_balanced_{size}_trainseed_{seed}_{suite}.json"
                if seed == 42:
                    path = metrics_dir / f"scale_balanced_{size}_{suite}.json"
                if path.exists():
                    report = json.loads(path.read_text(encoding="utf-8"))
                    values.append({"seed": seed, "task_success": report["task_success_rate"], "normalized_task_success": report.get("normalized_task_success_rate")})
            scores = [value["task_success"] for value in values]
            row[suite] = {"runs": values, "mean_task_success": sum(scores) / len(scores) if scores else None, "range": [min(scores), max(scores)] if scores else None}
        summary["sizes"][str(size)] = row
    out = metrics_dir / "scale_balanced_multiseed_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
