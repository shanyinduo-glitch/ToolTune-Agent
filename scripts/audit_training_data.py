"""Audit JSONL data with duplicate summaries and explicit review gates."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from difflib import SequenceMatcher
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import read_jsonl

def normalize_prompt(value: str) -> str:
    return " ".join(value.casefold().split())

def action_issues(action: Any, tools: list[dict[str, Any]]) -> list[str]:
    if not isinstance(action, dict): return ["missing_target_action"]
    name, arguments = action.get("name"), action.get("arguments")
    by_name = {t.get("function", {}).get("name"): t.get("function", {}) for t in tools}
    if name not in by_name: return ["target_tool_missing_from_schema"]
    if not isinstance(arguments, dict): return ["target_arguments_not_object"]
    params = by_name[name].get("parameters", {}); props = params.get("properties", {}); issues = []
    if set(params.get("required", [])) - arguments.keys(): issues.append("missing_required_arguments")
    if params.get("additionalProperties") is False and arguments.keys() - props.keys(): issues.append("unknown_arguments")
    for key, value in arguments.items():
        expected = props.get(key, {}).get("type")
        checks = {"string": lambda x: isinstance(x, str), "boolean": lambda x: isinstance(x, bool), "integer": lambda x: isinstance(x, int) and not isinstance(x, bool), "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool), "object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list)}
        if expected in checks and not checks[expected](value): issues.append(f"argument_type_mismatch:{key}")
    return issues

def audit_file(path: Path, threshold: float) -> dict[str, Any]:
    records = read_jsonl(path); prompt_map = defaultdict(list); action_map = defaultdict(list)
    counters = {key: Counter() for key in ("task_types", "sources", "splits", "difficulty", "families")}; issue_counts = Counter(); rows = []
    for record in records:
        prompt = next((m.get("content", "") for m in record.messages if m.get("role") == "user"), "")
        normalized = normalize_prompt(prompt); prompt_map[normalized].append(record.example_id)
        action_map[json.dumps(record.target_action, sort_keys=True, ensure_ascii=False)].append(record.example_id)
        meta = record.metadata
        counters["task_types"][record.task_type] += 1; counters["sources"][meta.get("source", "unspecified")] += 1; counters["splits"][meta.get("split", "unspecified")] += 1; counters["difficulty"][meta.get("difficulty", "unspecified")] += 1; counters["families"][meta.get("prompt_family", "unspecified")] += 1
        issues = action_issues(record.target_action, record.tools)
        if not prompt.strip(): issues.append("empty_user_prompt")
        if meta.get("quality_flags"): issues.append("unresolved_quality_flags")
        if meta.get("quality_gate") == "schema_and_format" and meta.get("semantic_review") != "passed": issues.append("semantic_review_required")
        if meta.get("training_allowed") is False: issues.append("training_not_allowed")
        if meta.get("semantic_review") not in (None, "passed"): issues.append("semantic_review_not_passed")
        issue_counts.update(issues); rows.append((record.example_id, normalized, issues))
    duplicates = {p: ids for p, ids in prompt_map.items() if len(ids) > 1}
    repeated_targets = {a: ids for a, ids in action_map.items() if len(ids) > 1}
    # Near-duplicates are a diagnostic signal. Template variation with a changed
    # entity is expected in synthetic pilots, so it is not a trainability gate.
    near_count = 0; near_examples = []
    prompts = [(i, p) for i, p, _ in rows if p]
    for idx, (left_id, left) in enumerate(prompts):
        for right_id, right in prompts[idx + 1:]:
            score = SequenceMatcher(None, left, right).ratio()
            if score >= threshold and left != right:
                near_count += 1
                if len(near_examples) < 20: near_examples.append({"left": left_id, "right": right_id, "similarity": round(score, 4)})
    return {"path": str(path), "records": len(records), "unique_prompts": len(prompt_map), "duplicate_prompt_groups": len(duplicates), "duplicate_prompt_examples": dict(list(duplicates.items())[:20]), "near_duplicate_pair_count": near_count, "near_duplicate_examples": near_examples, "repeated_target_groups": len(repeated_targets), "task_types": dict(sorted(counters["task_types"].items())), "sources": dict(sorted(counters["sources"].items())), "splits": dict(sorted(counters["splits"].items())), "difficulty": dict(sorted(counters["difficulty"].items())), "prompt_families": dict(sorted(counters["families"].items())), "quality_issue_counts": dict(sorted(issue_counts.items())), "records_with_quality_issues": sum(bool(x[2]) for x in rows), "trainable_by_static_gates": not issue_counts and not duplicates}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("inputs", nargs="+"); parser.add_argument("--near-duplicate-threshold", type=float, default=.92); parser.add_argument("--output", default=str(ROOT / "outputs/reports/data_audit.json")); args = parser.parse_args()
    report = {"near_duplicate_threshold": args.near_duplicate_threshold, "files": [audit_file(Path(name), args.near_duplicate_threshold) for name in args.inputs]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
