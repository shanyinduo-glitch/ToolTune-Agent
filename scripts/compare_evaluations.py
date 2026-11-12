"""Create a compact Base-vs-LoRA comparison from evaluator JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row(label: str, data: dict) -> dict:
    return {
        "model": label,
        "suite": data.get("suite"),
        "cases": data.get("num_cases", 0),
        "parse_rate": data.get("parse_rate", 0.0),
        "tool_accuracy": data.get("tool_accuracy", 0.0),
        "argument_accuracy": data.get("argument_accuracy", 0.0),
        "normalized_task_success_rate": data.get("normalized_task_success_rate", 0.0),
        "task_success_rate": data.get("task_success_rate", 0.0),
        "failure_counts": data.get("failure_counts", {}),
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-orders", required=True)
    parser.add_argument("--adapter-orders", required=True)
    parser.add_argument("--base-products", required=True)
    parser.add_argument("--adapter-products", required=True)
    parser.add_argument("--base-rollouts", required=True)
    parser.add_argument("--adapter-rollouts", required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs/metrics/challenge_comparison.json"))
    parser.add_argument("--markdown", default=str(ROOT / "outputs/reports/challenge_comparison.md"))
    args = parser.parse_args()

    base_order, tuned_order = load(args.base_orders), load(args.adapter_orders)
    base_product, tuned_product = load(args.base_products), load(args.adapter_products)
    base_rollouts, tuned_rollouts = load(args.base_rollouts), load(args.adapter_rollouts)
    metrics = ("parse_rate", "tool_accuracy", "argument_accuracy", "normalized_task_success_rate", "task_success_rate")
    groups = {}
    for name, base_data, tuned_data in (("orders", base_order, tuned_order), ("products", base_product, tuned_product)):
        base_row, tuned_row = row("base", base_data), row("decision_lora", tuned_data)
        base_cases = {case["case_id"]: case for case in base_data["cases"]}
        tuned_cases = {case["case_id"]: case for case in tuned_data["cases"]}
        paired = []
        for case_id in base_cases:
            if case_id not in tuned_cases:
                continue
            paired.append({"case_id": case_id, "base_success": base_cases[case_id]["task_success"], "adapter_success": tuned_cases[case_id]["task_success"], "base_failure": base_cases[case_id]["failure_reason"], "adapter_failure": tuned_cases[case_id]["failure_reason"]})
        groups[name] = {"base": base_row, "adapter": tuned_row, "delta": {key: tuned_row[key] - base_row[key] for key in metrics}, "paired_cases": paired, "wins": sum(x["adapter_success"] and not x["base_success"] for x in paired), "regressions": sum(x["base_success"] and not x["adapter_success"] for x in paired), "ties": sum(x["base_success"] == x["adapter_success"] for x in paired)}
    all_base = [*base_order["cases"], *base_product["cases"], *base_rollouts["cases"]]
    all_tuned = [*tuned_order["cases"], *tuned_product["cases"], *tuned_rollouts["cases"]]
    combined = {"cases": len(all_base), "base_task_success_rate": sum(x["task_success"] for x in all_base) / len(all_base), "adapter_task_success_rate": sum(x["task_success"] for x in all_tuned) / len(all_tuned), "base_parse_rate": sum(x["parsed"] for x in [*base_order["cases"], *base_product["cases"]]) / 24, "adapter_parse_rate": sum(x["parsed"] for x in [*tuned_order["cases"], *tuned_product["cases"]]) / 24}
    rollout_comparison = {}
    for category in ("multi_step", "recovery"):
        base_items = [item for item in base_rollouts["cases"] if item["category"] == category]
        tuned_items = [item for item in tuned_rollouts["cases"] if item["category"] == category]
        rollout_comparison[category] = {"cases": len(base_items), "base_passed": sum(item["task_success"] for item in base_items), "adapter_passed": sum(item["task_success"] for item in tuned_items)}
    comparison = {"groups": groups, "rollouts": rollout_comparison, "combined": combined, "interpretation": "Fixed deterministic challenge suites; these rates describe this suite and are not population estimates."}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Challenge Set Comparison", "", "This fixed challenge set contains paraphrases, mixed Chinese and English prompts, unknown IDs, confirmation boundaries, zero-stock products, and casing variants.", "", "| Suite | Model | Cases | Parse | Tool | Exact args | Normalized success | Exact success | Adapter wins | Regressions |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in groups.items():
        for label in ("base", "adapter"):
            item = group[label]
            wins = group["wins"] if label == "adapter" else "-"
            regressions = group["regressions"] if label == "adapter" else "-"
            lines.append(f"| {name} | {label} | {item['cases']} | {pct(item['parse_rate'])} | {pct(item['tool_accuracy'])} | {pct(item['argument_accuracy'])} | {pct(item['normalized_task_success_rate'])} | {pct(item['task_success_rate'])} | {wins} | {regressions} |")
    lines.extend(["", "## Multi-step and Recovery", "", "| Scenario | Cases | Base passed | Decision LoRA passed |", "|---|---:|---:|---:|"])
    for category, item in rollout_comparison.items():
        lines.append(f"| {category.replace('_', ' ')} | {item['cases']} | {item['base_passed']} | {item['adapter_passed']} |")
    lines.extend(["", f"Across all {combined['cases']} cases, exact task success was {pct(combined['base_task_success_rate'])} for base and {pct(combined['adapter_task_success_rate'])} for decision LoRA; parse rates on the 24 single-call cases were {pct(combined['base_parse_rate'])} and {pct(combined['adapter_parse_rate'])}.", "", "The suites are fixed and deterministic. Their rates describe these cases and should not be read as population estimates. Per-case outcomes and failure categories are in `outputs/metrics/challenge_comparison.json`.", ""])
    md = Path(args.markdown)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out), "markdown": str(md), "combined": combined}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
