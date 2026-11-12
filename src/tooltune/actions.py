"""Parse the small set of action formats emitted by causal language models."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_action(text: str) -> dict[str, Any] | None:
    """Return ``{name, arguments}`` from JSON, fenced JSON, or Qwen-style text."""
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, re.DOTALL))
    for candidate in candidates:
        candidate = candidate.strip()
        for value in _json_objects(candidate):
            action = _normalise(value)
            if action:
                return action
    match = re.search(r"(?:function|tool)\s*[:：]?\s*([A-Za-z_]\w*)\s*\((.*?)\)", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            arguments = json.loads("{" + match.group(2) + "}")
        except json.JSONDecodeError:
            return None
        return {"name": match.group(1), "arguments": arguments}
    return None


def _json_objects(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield value


def _normalise(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if "name" in value and "arguments" in value:
        arguments = value["arguments"]
    elif value.get("type") == "function":
        value = value.get("function", {})
        arguments = value.get("arguments", {})
    else:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict) or not isinstance(value.get("name"), str):
        return None
    return {"name": value["name"], "arguments": arguments}
