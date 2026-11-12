"""Build a deterministic, varied local SFT set for confirmation-aware tool use."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.environment import OrderToolEnvironment, ProductToolEnvironment
from tooltune.schema import ToolExample, write_jsonl


def make_example(index, prompt, name, arguments, result, difficulty):
    env = OrderToolEnvironment()
    messages = [
        {"role": "system", "content": "You are an order assistant. Use tools and follow confirmation and shipping rules."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]},
        {"role": "tool", "content": json.dumps(result, ensure_ascii=False)},
    ]
    return ToolExample(f"aug_{index:03d}", "confirmation" if name == "cancel_order" else "single_tool", messages, env.tool_definitions(), {"name": name, "arguments": arguments}, {"split": "train", "difficulty": difficulty, "source": "deterministic_augmentation_v1"})


def main():
    rows = []
    prompts = [(p, "A9999" if "9999" in p else ("A1002" if "1002" in p else "A1001")) for p in ["查询订单 A1001 的状态。", "订单 A1001 现在是什么状态？", "帮我看看 A1001。", "请告诉我订单 A1001 的信息。", "查询订单 A1002 的状态。", "订单 A1002 发货了吗？", "帮我看看 A1002。", "请告诉我订单 A1002 的信息。", "查找订单 A9999。", "订单 A9999 存在吗？", "查询不存在的订单 A9999。", "帮我确认 A9999 的订单状态。"]]
    for i, (prompt, order_id) in enumerate(prompts):
        result = {"ok": False, "error": "order_not_found", "order_id": order_id} if order_id == "A9999" else {"ok": True, "order_id": order_id, "order": OrderToolEnvironment().orders[order_id]}
        rows.append(make_example(i, prompt, "get_order", {"order_id": order_id}, result, "medium" if order_id == "A9999" else "easy"))
    for i, prompt in enumerate(["我想取消订单 A1001。", "请帮我取消 A1001。", "订单 A1001 不要了。", "可以取消订单 A1001 吗？", "我需要取消 A1001，请先处理。", "请申请取消订单 A1001。", "取消订单 A1001。", "帮我撤销 A1001。"], len(rows)):
        rows.append(make_example(i, prompt, "cancel_order", {"order_id": "A1001", "confirmed": False}, {"ok": False, "error": "confirmation_required", "order_id": "A1001"}, "medium"))
    for i, prompt in enumerate(["取消已发货订单 A1002。", "请撤销已经发货的 A1002。", "A1002 已发货还能取消吗？", "帮我取消 A1002。"], len(rows)):
        rows.append(make_example(i, prompt, "cancel_order", {"order_id": "A1002", "confirmed": False}, {"ok": False, "error": "already_shipped", "order_id": "A1002"}, "hard"))
    tools = ProductToolEnvironment().tool_definitions()
    for i, prompt in enumerate(["Find the wireless keyboard and check its inventory.", "Search for the keyboard, then check stock.", "Can you find the wireless keyboard and tell me its inventory?", "Look up the keyboard product and verify available stock."], len(rows)):
        messages = [
            {"role": "system", "content": "You are a shopping assistant. After search results, call check_inventory with the returned product_id."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "search_product", "arguments": json.dumps({"query": "wireless keyboard"})}}]},
            {"role": "tool", "content": json.dumps({"ok": True, "done": False, "results": [{"product_id": "keyboard", "name": "wireless keyboard", "stock": 12}]} )},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "check_inventory", "arguments": json.dumps({"product_id": "keyboard"})}}]},
            {"role": "tool", "content": json.dumps({"ok": True, "done": True, "product_id": "keyboard", "stock": 12})},
        ]
        rows.append(ToolExample(f"aug_{i:03d}", "multi_step", messages, tools, {"name": "check_inventory", "arguments": {"product_id": "keyboard"}}, {"split": "train", "difficulty": "hard", "source": "deterministic_augmentation_v1"}))
    output = ROOT / "data/splits/train_augmented.jsonl"
    write_jsonl(output, rows)
    print(f"wrote={output} records={len(rows)}")


if __name__ == "__main__":
    main()
