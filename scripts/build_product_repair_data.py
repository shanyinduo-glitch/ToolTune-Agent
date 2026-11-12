"""Build a small, executable multilingual product repair set.

These hand-authored examples target Chinese catalog aliases and search-then-
inventory grounding. They are kept separate from the original 500-record pilot.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.environment import ProductToolEnvironment
from tooltune.schema import ToolExample, read_jsonl, write_jsonl

PRODUCTS = [
    ("keyboard", "wireless keyboard", "無線鍵盤", "无线键盘"),
    ("microphone", "usb microphone", "麥克風", "麦克风"),
    ("mouse", "wireless mouse", "無線滑鼠", "无线鼠标"),
    ("webcam", "usb webcam", "USB 攝影機", "USB摄像头"),
    ("stand", "laptop stand", "筆記型電腦支架", "笔记本支架"),
]

def call(name: str, args: dict) -> dict:
    return {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}

def example(example_id: str, task: str, prompt: str, actions: list[tuple[str, dict]], family: str) -> ToolExample:
    env = ProductToolEnvironment()
    messages = [{"role": "system", "content": "Use the available catalog tools. For a multi-step request, search first and use the product_id returned by the search."}, {"role": "user", "content": prompt}]
    for name, args in actions:
        result = env.execute(name, args)
        if name == "search_product" and not result.get("results"):
            raise ValueError(f"repair search did not match catalog: {example_id}")
        messages.extend([call(name, args), {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)}])
    last_name, last_args = actions[-1]
    return ToolExample(example_id, task, messages, env.tool_definitions(), {"name": last_name, "arguments": last_args}, {
        "split": "repair_train", "source": "curated_multilingual_product_repair_v1",
        "prompt_family": family, "difficulty": "medium",
        "validation": "executed_against_default_product_environment",
        "semantic_review": "passed", "training_allowed": True,
    })

def build() -> list[ToolExample]:
    rows = []
    for product_id, canonical, _traditional, alias in PRODUCTS:
        rows.append(example(f"repair_{product_id}_stock_zh", "single_tool", f"請查一下{alias}目前還有多少庫存。", [("check_inventory", {"product_id": product_id})], "zh_alias_inventory"))
        rows.append(example(f"repair_{product_id}_stock_zh_context", "single_tool", f"我想買{alias}，先幫我確認可售數量。", [("check_inventory", {"product_id": product_id})], "zh_context_inventory"))
        rows.append(example(f"repair_{product_id}_stock_en", "single_tool", f"Would you verify how many {canonical}s remain in stock?", [("check_inventory", {"product_id": product_id})], "en_paraphrase_inventory"))
        rows.append(example(f"repair_{product_id}_search_zh", "single_tool", f"在商品目錄中找出{alias}。", [("search_product", {"query": canonical})], "zh_alias_search"))
        rows.append(example(f"repair_{product_id}_multi_zh", "multi_step", f"先搜尋{alias}，再根據搜尋結果查出庫存數量。", [("search_product", {"query": canonical}), ("check_inventory", {"product_id": product_id})], "zh_alias_multistep"))
        rows.append(example(f"repair_{product_id}_multi_en", "multi_step", f"Find the {canonical} listing and then check its current stock.", [("search_product", {"query": canonical}), ("check_inventory", {"product_id": product_id})], "en_multistep"))
    return rows

def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data/processed/product_repair_30.jsonl"))
    parser.add_argument("--merged-train", default=str(ROOT / "data/splits/pilot_500_reviewed/train_repair_416.jsonl"))
    args = parser.parse_args()
    additions = build(); write_jsonl(args.output, additions)
    base = read_jsonl(ROOT / "data/splits/pilot_500_reviewed/train.jsonl")
    ids = {x.example_id for x in base}
    if ids.intersection(x.example_id for x in additions): raise ValueError("repair IDs overlap with base training IDs")
    write_jsonl(args.merged_train, [*base, *additions])
    print(json.dumps({"repair_records": len(additions), "combined_train_records": len(base) + len(additions), "combined_decision_points": sum(len(m.get("tool_calls", [])) for row in [*base, *additions] for m in row.messages if m.get("role") == "assistant"), "training_allowed": all(x.metadata.get("training_allowed") is True for x in additions)}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
