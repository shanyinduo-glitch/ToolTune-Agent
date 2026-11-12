"""Strict JSONL schema used by the first tool-use experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class ToolExample:
    example_id: str
    task_type: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    target_action: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.example_id or not isinstance(self.example_id, str):
            raise ValueError("example_id must be a non-empty string")
        if not self.task_type or not isinstance(self.task_type, str):
            raise ValueError("task_type must be a non-empty string")
        if not self.messages:
            raise ValueError("messages must not be empty")
        for index, message in enumerate(self.messages):
            if not isinstance(message, dict) or message.get("role") not in ROLES:
                raise ValueError(f"messages[{index}] has an invalid role")
            if "content" not in message and "tool_calls" not in message:
                raise ValueError(f"messages[{index}] needs content or tool_calls")
        for tool in self.tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("tools must use the OpenAI function schema")
            function = tool.get("function", {})
            if not function.get("name") or "parameters" not in function:
                raise ValueError("each function tool needs name and parameters")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "example_id": self.example_id,
            "task_type": self.task_type,
            "messages": self.messages,
            "tools": self.tools,
            "target_action": self.target_action,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ToolExample":
        item = cls(
            example_id=value.get("example_id", ""),
            task_type=value.get("task_type", ""),
            messages=value.get("messages", []),
            tools=value.get("tools", []),
            target_action=value.get("target_action"),
            metadata=value.get("metadata", {}),
        )
        item.validate()
        return item


def write_jsonl(path: str | Path, examples: Iterable[ToolExample]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[ToolExample]:
    items: list[ToolExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                items.append(ToolExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"invalid JSONL record at line {line_number}: {exc}") from exc
    return items
