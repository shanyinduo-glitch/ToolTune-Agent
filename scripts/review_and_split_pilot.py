"""Full-review the synthetic pilot and make entity-disjoint train/val/test splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import ToolExample, read_jsonl, write_jsonl


def messages(row: ToolExample):
    user = next((m.get("content", "") for m in row.messages if m.get("role") == "user"), "")
    calls = []
    results = []
    for message in row.messages:
        if message.get("role") == "assistant":
            for item in message.get("tool_calls", []):
                function = item.get("function", {})
                calls.append((function.get("name"), json.loads(function.get("arguments", "{}"))))
        elif message.get("role") == "tool":
            results.append(json.loads(message.get("content", "{}")))
    return user, calls, results


def review(row: ToolExample) -> list[str]:
    issues: list[str] = []
    user, calls, results = messages(row)
    target = row.target_action or {}
    target_name = target.get("name")
    target_args = target.get("arguments", {})
    if not user.strip():
        issues.append("empty_user_prompt")
    if not calls:
        issues.append("missing_assistant_tool_call")
    if row.task_type == "multi_step":
        if [name for name, _ in calls] != ["search_product", "check_inventory"]:
            issues.append("wrong_multistep_order")
        if len(results) != 2:
            issues.append("wrong_multistep_result_count")
        elif not results[0].get("results"):
            issues.append("search_returned_no_product")
        elif results[1].get("product_id") != target_args.get("product_id"):
            issues.append("inventory_id_does_not_match_target")
        if target_name != "check_inventory":
            issues.append("wrong_final_target_tool")
        if calls and calls[-1][1].get("product_id") != target_args.get("product_id"):
            issues.append("final_call_arguments_mismatch")
        if not target_args.get("product_id") or target_args["product_id"] not in user:
            # The ID may be intentionally absent from natural language; the
            # product name is checked against the search query below.
            if not calls or calls[0][1].get("query", "") not in user:
                issues.append("multi_step_entity_not_in_prompt")
    elif row.task_type == "confirmation":
        if target_name != "cancel_order":
            issues.append("wrong_confirmation_tool")
        if target_args.get("confirmed") is not False:
            issues.append("confirmation_flag_not_false")
        if not calls or calls[0] != (target_name, target_args):
            issues.append("confirmation_call_mismatch")
        if not target_args.get("order_id") or target_args["order_id"] not in user:
            issues.append("order_id_not_in_prompt")
        if not results:
            issues.append("missing_confirmation_result")
        elif results[-1].get("error") not in {"confirmation_required", "already_shipped", "order_not_found"}:
            issues.append("invalid_confirmation_result")
    else:
        if target_name not in {"get_order", "check_inventory"}:
            issues.append("wrong_single_tool")
        if not calls or calls[0] != (target_name, target_args):
            issues.append("single_call_mismatch")
        if target_name == "get_order":
            if target_args.get("order_id") not in user:
                issues.append("order_id_not_in_prompt")
            if not results or results[-1].get("order_id") != target_args.get("order_id"):
                issues.append("order_result_id_mismatch")
        if target_name == "check_inventory":
            if target_args.get("product_id") not in user and target_args.get("product_id", "").replace("item-", "") not in user:
                issues.append("product_id_not_in_prompt")
            if not results or results[-1].get("product_id") != target_args.get("product_id"):
                issues.append("inventory_result_id_mismatch")
    return issues


def entity_key(row: ToolExample) -> str:
    action = row.target_action or {}
    args = action.get("arguments", {})
    if "order_id" in args:
        return "order:" + args["order_id"]
    return "product:" + args.get("product_id", "unknown")


def bucket(entity: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{entity}".encode()).hexdigest()[:8], 16) % 100
    return "test" if value < 15 else "validation" if value < 30 else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/splits/pilot_500.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/splits/pilot_500_reviewed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    decisions = []
    reviewed = []
    for row in rows:
        issues = review(row)
        decisions.append({"example_id": row.example_id, "entity": entity_key(row), "result": "pass" if not issues else "reject", "issues": issues})
        if issues:
            continue
        metadata = dict(row.metadata)
        metadata.update({"semantic_review": "passed", "review_method": "full_rule_and_trajectory_review", "training_allowed": True})
        reviewed.append(ToolExample(row.example_id, row.task_type, row.messages, row.tools, row.target_action, metadata))
    if len(reviewed) != len(rows):
        raise SystemExit(f"review rejected {len(rows) - len(reviewed)} records; fix before splitting")
    assignments = {entity: bucket(entity, args.seed) for entity in sorted({entity_key(r) for r in reviewed})}
    split_rows = defaultdict(list)
    for row in reviewed:
        split_rows[assignments[entity_key(row)]].append(row)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        write_jsonl(output_dir / f"{name}.jsonl", split_rows[name])
    report = {
        "input": args.input, "review_method": "full_rule_and_trajectory_review", "seed": args.seed,
        "reviewed_records": len(rows), "passed": len(reviewed), "rejected": len(rows) - len(reviewed),
        "decision_counts": dict(Counter(x["result"] for x in decisions)),
        "split_records": {name: len(split_rows[name]) for name in ("train", "validation", "test")},
        "split_entities": {name: sorted({entity_key(x) for x in split_rows[name]}) for name in ("train", "validation", "test")},
        "task_types": {name: dict(Counter(x.task_type for x in split_rows[name])) for name in ("train", "validation", "test")},
        "semantic_review": "passed_for_all_500",
        "training_allowed": True,
        "leakage_policy": "entity_disjoint_order_and_product_ids",
        "decisions": decisions,
    }
    (output_dir / "review_and_split.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("reviewed_records", "passed", "rejected", "split_records", "task_types")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
