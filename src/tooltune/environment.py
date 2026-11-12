"""Deterministic order environment with explicit tool-call validation."""

from __future__ import annotations

import copy
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Any, Callable


def validate_call(definitions, name, arguments):
    spec = next((t['function']['parameters'] for t in definitions if t['function']['name'] == name), None)
    if spec is None:
        return {'ok': False, 'error': 'unknown_tool', 'tool': name}
    if not isinstance(arguments, dict):
        return {'ok': False, 'error': 'invalid_arguments', 'message': 'arguments must be an object'}
    required = set(spec.get('required', []))
    props = spec['properties']
    if not required.issubset(arguments) or set(arguments) - set(props):
        return {'ok': False, 'error': 'invalid_arguments', 'message': 'missing or unexpected fields'}
    for key, value in arguments.items():
        typ = props[key]['type']
        if (typ == 'string' and (not isinstance(value, str) or not value.strip())) or (typ == 'boolean' and type(value) is not bool):
            return {'ok': False, 'error': 'invalid_arguments', 'message': f'{key} must be {typ} (nonempty for strings)'}
    return None


@dataclass
class OrderToolEnvironment:
    orders: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "A1001": {"status": "processing", "item": "wireless keyboard", "total": 49.99},
        "A1002": {"status": "shipped", "item": "usb microphone", "total": 79.00},
        "A2101": {"status": "processing", "item": "wireless mouse", "total": 25.50},
        "A2102": {"status": "shipped", "item": "usb webcam", "total": 59.00},
        "A2103": {"status": "processing", "item": "mechanical keyboard", "total": 88.00},
        "A2104": {"status": "delivered", "item": "desk lamp", "total": 32.00},
        "A2105": {"status": "processing", "item": "laptop stand", "total": 41.00},
    })
    trace: list[dict[str, Any]] = field(default_factory=list)
    pending_cancellation: str | None = None
    session_id: str = field(default_factory=lambda: uuid4().hex)
    _authorized_order: str | None = field(default=None, init=False, repr=False)

    def confirm_cancellation(self, order_id: str, session_id: str, approved: bool) -> bool:
        """Trusted UI/controller event, deliberately absent from model tool definitions."""
        if session_id != self.session_id or order_id != self.pending_cancellation or type(approved) is not bool:
            return False
        self._authorized_order = order_id if approved else None
        if not approved:
            self.pending_cancellation = None
        return approved

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "get_order", "description": "Get order status.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "cancel_order", "description": "Cancel an order after confirmation.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "confirmed": {"type": "boolean"}}, "required": ["order_id", "confirmed"], "additionalProperties": False}}},
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {"get_order": self._get_order, "cancel_order": self._cancel_order}
        invalid = validate_call(self.tool_definitions(), name, arguments)
        if invalid:
            result = invalid
        elif name not in handlers:
            result = {"ok": False, "error": "unknown_tool", "tool": name}
        else:
            try:
                result = handlers[name](arguments)
            except (KeyError, TypeError, ValueError) as exc:
                result = {"ok": False, "error": "invalid_arguments", "message": str(exc)}
        self.trace.append({"tool": name, "arguments": copy.deepcopy(arguments), "result": copy.deepcopy(result)})
        return result

    def _get_order(self, args: dict[str, Any]) -> dict[str, Any]:
        order_id = args["order_id"]
        if not isinstance(order_id, str):
            raise ValueError("order_id must be a string")
        order = self.orders.get(order_id)
        return {"ok": True, "order_id": order_id, "order": copy.deepcopy(order)} if order else {"ok": False, "error": "order_not_found", "order_id": order_id}

    def _cancel_order(self, args: dict[str, Any]) -> dict[str, Any]:
        order_id, confirmed = args["order_id"], args["confirmed"]
        if not isinstance(order_id, str) or not isinstance(confirmed, bool):
            raise ValueError("order_id must be string and confirmed must be boolean")
        if order_id not in self.orders:
            return {"ok": False, "error": "order_not_found", "order_id": order_id}
        status = self.orders[order_id]["status"]
        if status != "processing":
            error = "already_shipped" if status in {"shipped", "delivered"} else "not_cancellable"
            return {"ok": False, "error": error, "order_id": order_id}
        if not confirmed:
            self._authorized_order = None
            self.pending_cancellation = order_id
            return {"ok": False, "error": "confirmation_required", "order_id": order_id}
        if self.pending_cancellation != order_id or self._authorized_order != order_id:
            self.pending_cancellation = order_id
            return {"ok": False, "error": "confirmation_required", "order_id": order_id}
        self.orders[order_id]["status"] = "cancelled"
        self.pending_cancellation = None
        self._authorized_order = None
        return {"ok": True, "order_id": order_id, "status": "cancelled"}

    def reset_trace(self) -> None:
        self.trace.clear()


@dataclass
class ProductToolEnvironment:
    products: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "keyboard": {"name": "wireless keyboard", "price": 49.99, "stock": 12},
        "microphone": {"name": "usb microphone", "price": 79.0, "stock": 0},
        "mouse": {"name": "wireless mouse", "price": 25.5, "stock": 7},
        "webcam": {"name": "usb webcam", "price": 59.0, "stock": 3},
        "stand": {"name": "laptop stand", "price": 41.0, "stock": 0},
    })
    trace: list[dict[str, Any]] = field(default_factory=list)

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "search_product", "description": "Search a product by keyword.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "check_inventory", "description": "Check product stock.", "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"], "additionalProperties": False}}},
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        invalid = validate_call(self.tool_definitions(), name, arguments)
        if invalid:
            result = invalid
        elif name == "search_product":
            query = arguments.get("query")
            matches = [{"product_id": key, **value} for key, value in self.products.items() if isinstance(query, str) and query.lower() in value["name"].lower()]
            result = {"ok": True, "done": False, "results": matches}
        elif name == "check_inventory":
            product_id = arguments.get("product_id")
            product = self.products.get(product_id)
            result = {"ok": True, "done": True, "product_id": product_id, "stock": product["stock"]} if product else {"ok": False, "done": True, "error": "product_not_found", "product_id": product_id}
        else:
            result = {"ok": False, "error": "unknown_tool", "tool": name}
        self.trace.append({"tool": name, "arguments": copy.deepcopy(arguments), "result": copy.deepcopy(result)})
        return result
