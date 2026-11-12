"""Audit and optionally convert the public Hermes single-turn dataset.

This script deliberately keeps the dataset separate from the training splits. It
accepts only records with a usable system/user conversation, an OpenAI-style
tool list, and JSON tool calls that can be parsed without guessing.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import ToolExample, write_jsonl

OPEN_CALL_RE = re.compile(r"<tool_call>\s*", re.S)
END_CALL = "</tool_call>"


def parse_tools(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("tools is not a list")
    return value


def parse_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in OPEN_CALL_RE.finditer(text or ""):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text) or text[start] != "{":
            raise ValueError("malformed tool call JSON")
        depth = 0
        in_string = False
        escaped = False
        end = None
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
        if end is None:
            raise ValueError("malformed tool call JSON")
        closing = end
        while closing < len(text) and text[closing].isspace():
            closing += 1
        if not text.startswith(END_CALL, closing):
            raise ValueError("malformed tool call wrapper")
        item = json.loads(text[start:end])
        if not isinstance(item, dict) or not item.get("name") or not isinstance(item.get("arguments"), dict):
            raise ValueError("invalid tool call shape")
        calls.append(item)
    if not calls:
        raise ValueError("no parseable tool calls")
    return calls


def convert(item: dict[str, Any], index: int) -> ToolExample:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("conversations is not a list")
    human = next((m for m in conversations if m.get("from") == "human"), None)
    assistant = next((m for m in reversed(conversations) if m.get("from") == "gpt"), None)
    if not human or not assistant or not isinstance(human.get("value"), str) or not isinstance(assistant.get("value"), str):
        raise ValueError("missing human/gpt turn")
    tools = parse_tools(item.get("tools", []))
    calls = parse_calls(assistant["value"])
    available_tools = {tool["function"]["name"] for tool in tools}
    if any(call["name"] not in available_tools for call in calls):
        raise ValueError("tool call name is absent from tool schema")
    messages = [
        {"role": "system", "content": conversations[0].get("value", "") if conversations[0].get("from") == "system" else "You are a function calling assistant."},
        {"role": "user", "content": human["value"]},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in calls]},
    ]
    return ToolExample(
        example_id=f"hermes_singleturn_{index:05d}",
        task_type="multi_step" if len(calls) > 1 else "single_tool",
        messages=messages,
        tools=tools,
        target_action={"name": calls[-1]["name"], "arguments": calls[-1]["arguments"]},
        metadata={"split": "audit", "source": "NousResearch/hermes-function-calling-v1", "license": "Apache-2.0", "source_id": item.get("id"), "call_count": len(calls), "category": item.get("category"), "subcategory": item.get("subcategory")},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "outputs/cache/datasets/hermes-function-calling-v1/func-calling-singleturn.json"))
    parser.add_argument("--output", default=str(ROOT / "data/processed/hermes_audit.jsonl"))
    parser.add_argument("--limit", type=int, default=100, help="maximum converted records; use 0 for all")
    args = parser.parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    stats = Counter()
    rejection_types = Counter()
    rejected_samples: dict[str, list[dict[str, Any]]] = {}
    converted: list[ToolExample] = []
    for index, item in enumerate(raw):
        stats["total"] += 1
        try:
            example = convert(item, index)
        except ValueError as exc:
            message = str(exc)
            if message == "no parseable tool calls":
                category = "no_tool_call_response"
            elif "tool call name is absent" in message:
                category = "tool_name_not_in_schema"
            elif "tool call JSON" in message or "tool call wrapper" in message:
                answer = next((m.get("value", "") for m in reversed(item.get("conversations", [])) if m.get("from") == "gpt"), "")
                payload = answer.replace("\\n", "\n")
                payloads = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", payload, re.S)
                python_literal = False
                for candidate in payloads:
                    try:
                        parsed_literal = ast.literal_eval(candidate)
                        python_literal = isinstance(parsed_literal, dict)
                    except (ValueError, SyntaxError):
                        pass
                    if python_literal:
                        break
                if python_literal:
                    category = "python_literal_payload"
                elif re.search(r"'[^']+'\s*:", answer):
                    category = "python_single_quoted_payload"
                elif "<tool_call>" in answer:
                    category = "malformed_tool_call_wrapper_or_json"
                else:
                    category = "non_tool_text_or_format"
            else:
                category = "invalid_record"
            rejection_types[category] += 1
            stats["invalid"] += 1
            if len(rejected_samples.setdefault(category, [])) < 5:
                answer = next((m.get("value", "") for m in reversed(item.get("conversations", [])) if m.get("from") == "gpt"), "")
                names = re.findall(r'"name"\s*:\s*"([^"]+)"', answer)
                try:
                    available = {t["function"]["name"] for t in parse_tools(item.get("tools", []))}
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    available = set()
                rejected_samples[category].append({"index": index, "source_id": item.get("id"), "reason": message, "tool_names_visible": names[:5], "visible_names_in_schema": [name for name in names if name in available], "answer_excerpt": answer[:500]})
            continue
        except (KeyError, TypeError, json.JSONDecodeError):
            stats["invalid"] += 1
            continue
        stats["valid"] += 1
        stats[f"calls_{example.metadata['call_count']}"] += 1
        stats[f"task_{example.task_type}"] += 1
        if args.limit == 0 or len(converted) < args.limit:
            converted.append(example)
    write_jsonl(args.output, converted)
    report = {
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "license": "Apache-2.0",
        "top_level_fields": sorted(raw[0]) if raw else [],
        "stats": dict(stats),
        "rejection_types": dict(rejection_types),
        "rejected_sample_records": rejected_samples,
        "answer_types": {
            "single_tool": stats["task_single_tool"],
            "multi_step": stats["task_multi_step"],
        },
        "written": len(converted),
        "note": "Audit artifact only; not included in formal training splits.",
    }
    report_path = Path(args.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
