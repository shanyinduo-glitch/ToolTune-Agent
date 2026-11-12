"""Validate Hermes review candidates against tool schemas and simple quality gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import read_jsonl, write_jsonl


def type_ok(value: Any, schema: dict[str, Any]) -> bool:
    expected = schema.get("type")
    if expected == "string": return isinstance(value, str)
    if expected == "boolean": return isinstance(value, bool)
    if expected == "integer": return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number": return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array": return isinstance(value, list) and all(type_ok(x, schema.get("items", {})) for x in value)
    if expected == "object": return isinstance(value, dict)
    return True


def inspect(row) -> list[str]:
    issues: list[str] = []
    user = next((m.get("content", "") for m in row.messages if m.get("role") == "user"), "")
    if not user.strip(): issues.append("empty_user_prompt")
    action = row.target_action or {}
    tools = {t.get("function", {}).get("name"): t.get("function", {}) for t in row.tools}
    name = action.get("name")
    if name not in tools: issues.append("tool_missing_from_schema"); return issues
    params = tools[name].get("parameters", {})
    arguments = action.get("arguments")
    if not isinstance(arguments, dict): issues.append("arguments_not_object"); return issues
    required = params.get("required", [])
    for key in required:
        if key not in arguments: issues.append(f"missing_required:{key}")
    properties = params.get("properties", {})
    if params.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties: issues.append(f"unexpected_argument:{key}")
    for key, value in arguments.items():
        spec = properties.get(key, {})
        if not type_ok(value, spec): issues.append(f"type_mismatch:{key}")
        if "enum" in spec and value not in spec["enum"]: issues.append(f"enum_mismatch:{key}")
    if user.count("\n\n") >= 2: issues.append("multi_intent_prompt_review")
    for value in arguments.values():
        if isinstance(value, str) and len(value) > 2 and value.casefold() not in user.casefold():
            issues.append("argument_not_mentioned_review")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/processed/hermes_train_candidate_50.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/processed/hermes_quality_approved.jsonl"))
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    seen_ids, seen_prompts = set(), set()
    approved, records, reasons = [], [], Counter()
    for row in rows:
        issues = inspect(row)
        source_id = row.metadata.get("source_id")
        prompt = " ".join(next(m["content"] for m in row.messages if m.get("role") == "user").casefold().split())
        if source_id in seen_ids: issues.append("duplicate_source_id")
        if prompt in seen_prompts: issues.append("duplicate_prompt")
        seen_ids.add(source_id); seen_prompts.add(prompt)
        approved_flag = not any(issue.split(":", 1)[0] in {"tool_missing_from_schema", "arguments_not_object", "missing_required", "unexpected_argument", "type_mismatch", "enum_mismatch", "empty_user_prompt", "duplicate_source_id", "duplicate_prompt"} for issue in issues)
        records.append({"example_id": row.example_id, "source_id": source_id, "approved": approved_flag, "issues": issues})
        if approved_flag:
            row.metadata.update({"split": "train_candidate_quality_approved", "quality_gate": "schema_and_format"})
            approved.append(row)
        for issue in issues: reasons[issue.split(":", 1)[0]] += 1
    write_jsonl(args.output, approved)
    report = {"input": args.input, "output": args.output, "input_records": len(rows), "approved_records": len(approved), "rejected_records": len(rows) - len(approved), "reason_counts": dict(reasons), "records": records, "approval_scope": "automatic schema/format gate; semantic review flags remain visible"}
    Path(args.output).with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("input_records", "approved_records", "rejected_records", "reason_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
