"""Build a deterministic, executable 500-trajectory synthetic review pilot.

The resulting data is pending semantic review and is deliberately not marked
trainable. Each trajectory is validated against its own generated environment.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.environment import OrderToolEnvironment, ProductToolEnvironment
from tooltune.schema import ToolExample, write_jsonl

STATUS_TEMPLATES = [
    ("zh_direct", "请查询订单 {order_id} 的状态。"),
    ("zh_context", "帮我看看订单号 {order_id} 目前处理到哪一步了。"),
    ("en_direct", "What is the current status of order {order_id}?"),
]
CANCEL_TEMPLATES = [
    ("zh_direct", "我想取消订单 {order_id}，请先发起申请。"),
    ("zh_context", "订单编号 {order_id} 我不需要了，帮我申请取消并等待确认。"),
    ("en_direct", "Please start a cancellation request for order {order_id}, pending my confirmation."),
]
INVENTORY_TEMPLATES = [
    ("zh_direct", "查询商品 {name} 的库存数量。"),
    ("zh_context", "{name} 现在还有货吗？请核对库存。"),
    ("en_direct", "Check the available stock for {name}."),
    ("en_context", "How many units of {name} are currently available?"),
    ("zh_concise", "{name} 还有多少件？"),
]
MULTI_TEMPLATES = [
    ("zh_sequence", "先在目录中搜索 {name}，然后查询该商品的库存。"),
    ("en_sequence", "Find {name} in the catalog, then check its inventory."),
]


def call(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


def single(example_id: str, task: str, prompt: str, family: str, env, tool: str, args: dict, difficulty: str) -> ToolExample:
    result = env.execute(tool, args)
    if tool == "get_order" and result.get("order_id") != args["order_id"]:
        raise ValueError(f"order status validation failed: {example_id}")
    if tool == "check_inventory" and result.get("product_id") != args["product_id"]:
        raise ValueError(f"inventory validation failed: {example_id}")
    if tool == "cancel_order" and result.get("error") not in {"confirmation_required", "already_shipped", "order_not_found"}:
        raise ValueError(f"cancellation validation failed: {example_id}: {result}")
    return ToolExample(example_id, task, [
        {"role": "system", "content": "Use the available tools. Follow confirmation requirements."},
        {"role": "user", "content": prompt}, call(tool, args),
        {"role": "tool", "content": json.dumps(result, ensure_ascii=False)},
    ], env.tool_definitions(), {"name": tool, "arguments": args}, {
        "split": "pilot_review", "source": "synthetic_executable_pilot_v2",
        "prompt_family": family, "difficulty": difficulty,
        "validation": "executed_against_record_specific_environment",
        "semantic_review": "pending", "training_allowed": False,
    })


def multi(example_id: str, prompt: str, family: str, product_id: str, product: dict) -> ToolExample:
    env = ProductToolEnvironment(products={product_id: product})
    search_args, inventory_args = {"query": product["name"]}, {"product_id": product_id}
    found = env.execute("search_product", search_args)
    stock = env.execute("check_inventory", inventory_args)
    if not any(x["product_id"] == product_id for x in found["results"]) or stock.get("product_id") != product_id:
        raise ValueError(f"multi-step validation failed: {example_id}")
    return ToolExample(example_id, "multi_step", [
        {"role": "system", "content": "Search first, then use the returned product_id to check inventory."},
        {"role": "user", "content": prompt}, call("search_product", search_args),
        {"role": "tool", "content": json.dumps(found, ensure_ascii=False)},
        call("check_inventory", inventory_args),
        {"role": "tool", "content": json.dumps(stock, ensure_ascii=False)},
    ], env.tool_definitions(), {"name": "check_inventory", "arguments": inventory_args}, {
        "split": "pilot_review", "source": "synthetic_executable_pilot_v2",
        "prompt_family": family, "difficulty": "hard",
        "validation": "both_steps_executed_against_record_specific_environment",
        "semantic_review": "pending", "training_allowed": False,
    })


def build() -> list[ToolExample]:
    rows: list[ToolExample] = []
    # 60 unique order IDs: 45 known across three states, 15 unknown.
    orders = {}
    for i in range(45):
        status = ("processing", "shipped", "delivered")[i % 3]
        orders[f"P{i:04d}"] = {"status": status, "item": f"order item {i:04d}", "total": round(10 + i * 1.37, 2)}
    all_ids = list(orders) + [f"P{i:04d}" for i in range(45, 60)]
    n = 0
    for oid in all_ids:
        for family, template in STATUS_TEMPLATES:
            rows.append(single(f"pilot_{n:04d}", "single_tool", template.format(order_id=oid), family,
                OrderToolEnvironment(orders={k: dict(v) for k, v in orders.items()}), "get_order", {"order_id": oid}, "medium" if oid not in orders else "easy")); n += 1
    for oid in all_ids:
        for family, template in CANCEL_TEMPLATES:
            rows.append(single(f"pilot_{n:04d}", "confirmation", template.format(order_id=oid), family,
                OrderToolEnvironment(orders={k: dict(v) for k, v in orders.items()}), "cancel_order", {"order_id": oid, "confirmed": False}, "hard" if oid not in orders or orders.get(oid, {}).get("status") != "processing" else "medium")); n += 1

    # Put a trailing token after the numeric ID so substring search for item
    # 0001 cannot also match 00010 through 00019.
    products = {f"item-{i:04d}": {"name": f"catalog item code {i:04d} end", "price": round(3.5 + i * 1.19, 2), "stock": (0 if i % 4 == 0 else (i * 7) % 31)} for i in range(20)}
    for pid, product in products.items():
        for family, template in INVENTORY_TEMPLATES:
            rows.append(single(f"pilot_{n:04d}", "single_tool", template.format(name=product["name"]), family,
                ProductToolEnvironment(products={pid: dict(product)}), "check_inventory", {"product_id": pid}, "medium")); n += 1
    # 40 trajectories, each with two independently executed tool decisions.
    for pid, product in products.items():
        for family, template in MULTI_TEMPLATES:
            rows.append(multi(f"pilot_{n:04d}", template.format(name=product["name"]), family, pid, product)); n += 1
    if len(rows) != 500:
        raise AssertionError(f"expected 500 records, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data/splits/pilot_500.jsonl"))
    parser.add_argument("--manifest", default=str(ROOT / "data/splits/pilot_500.manifest.json"))
    parser.add_argument("--review-sample", default=str(ROOT / "data/splits/pilot_500_review_sample.jsonl"))
    parser.add_argument("--review-per-family", type=int, default=3)
    args = parser.parse_args()
    rows = build()
    write_jsonl(args.output, rows)
    families = Counter(x.metadata["prompt_family"] for x in rows)
    tasks = Counter(x.task_type for x in rows)
    decisions = sum(len(m.get("tool_calls", [])) for row in rows for m in row.messages if m["role"] == "assistant")
    manifest = {
        "records": len(rows), "assistant_decision_points": decisions,
        "task_types": dict(sorted(tasks.items())), "prompt_families": dict(sorted(families.items())),
        "unique_order_ids": 60, "known_orders": 45, "unknown_order_ids": 15,
        "unique_product_ids": 20, "source": "synthetic_executable_pilot_v2",
        "validation": "all target tool calls executed against deterministic record-specific environments",
        "semantic_review": "pending", "training_allowed": False,
        "review_instruction": "Review each sampled prompt, tool choice, arguments, environment response and tool chronology; approve explicitly before training.",
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sample = []
    seen = Counter()
    for row in rows:
        stratum = (row.task_type, row.metadata["prompt_family"])
        if seen[stratum] < args.review_per_family:
            sample.append(row); seen[stratum] += 1
    write_jsonl(args.review_sample, sample)
    print(json.dumps({"records": len(rows), "assistant_decision_points": decisions,
                      "task_types": dict(tasks), "prompt_families": dict(families),
                      "review_sample_records": len(sample), "training_allowed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
