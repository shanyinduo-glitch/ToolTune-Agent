"""Create a review-only JSONL candidate from Hermes Python-literal calls."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import ToolExample, write_jsonl


def clean_text(value: Any, counter: list[int]) -> Any:
    if isinstance(value, str):
        cleaned = value.encode("utf-8", errors="replace").decode("utf-8")
        if cleaned != value:
            counter[0] += 1
        return cleaned
    if isinstance(value, list):
        return [clean_text(item, counter) for item in value]
    if isinstance(value, dict):
        return {key: clean_text(item, counter) for key, item in value.items()}
    return value


def parse_payload(value: str) -> dict[str, Any]:
    start, end = value.find("<tool_call>"), value.find("</tool_call>")
    if start < 0 or end < 0:
        raise ValueError("missing tool-call wrapper")
    payload = value[start + len("<tool_call>"):end].strip().replace("\\n", "\n")
    parsed = ast.literal_eval(payload)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("arguments"), dict) or not parsed.get("name"):
        raise ValueError("invalid literal call shape")
    return parsed


def convert(item: dict[str, Any], index: int, replacements: list[int]) -> ToolExample:
    item = clean_text(item, replacements)
    conversations = item["conversations"]
    human = next(m for m in conversations if m.get("from") == "human")
    assistant = next(m for m in reversed(conversations) if m.get("from") == "gpt")
    calls = [clean_text(parse_payload(assistant["value"]), replacements)]
    tools = json.loads(item["tools"]) if isinstance(item.get("tools"), str) else item["tools"]
    tools = clean_text(tools, replacements)
    names = {tool["function"]["name"] for tool in tools}
    if calls[0]["name"] not in names:
        raise ValueError("tool name absent from schema")
    return ToolExample(
        example_id=f"hermes_literal_{index:05d}", task_type="single_tool", tools=tools,
        messages=[{"role": "system", "content": "You are a function calling assistant."}, {"role": "user", "content": human["value"]}, {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": calls[0]["name"], "arguments": json.dumps(calls[0]["arguments"], ensure_ascii=False)}}]}],
        target_action={"name": calls[0]["name"], "arguments": calls[0]["arguments"]},
        metadata={"split": "review_only", "source": "NousResearch/hermes-function-calling-v1", "license": "Apache-2.0", "source_id": item.get("id"), "category": item.get("category"), "subcategory": item.get("subcategory"), "conversion": "ast.literal_eval_then_json"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "outputs/cache/datasets/hermes-function-calling-v1/func-calling-singleturn.json"))
    parser.add_argument("--output", default=str(ROOT / "data/processed/hermes_literal_review.jsonl"))
    args = parser.parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    converted, rejected = [], Counter()
    replacements = [0]
    for index, item in enumerate(raw):
        try:
            converted.append(convert(item, index, replacements))
        except (KeyError, TypeError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
            text = str(exc)
            reason = "non_literal_expression" if "malformed node or string" in text else ("unterminated_string" if "unterminated string" in text else text)
            rejected[reason] += 1
    write_jsonl(args.output, converted)
    report = {"input": args.input, "output": args.output, "license": "Apache-2.0", "records": len(converted), "rejected": sum(rejected.values()), "rejection_reasons": dict(rejected), "unicode_replacements": replacements[0], "review_only": True, "not_in_training_splits": True}
    Path(args.output).with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
