"""Evaluate multi-step product tasks and one model-driven recovery path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.actions import parse_action
from tooltune.environment import ProductToolEnvironment
from tooltune.reproducibility import seed_everything
from tooltune.evaluation import fresh_stress_rollout_specs


CASES = [
    ("multi_keyboard", "Find the wireless keyboard and check how many are in stock.", [("search_product", {"query": "wireless keyboard"}), ("check_inventory", {"product_id": "keyboard"})], 12, False),
    ("multi_mouse_cn", "找到无线鼠标，再查一下库存数量。", [("search_product", {"query": "wireless mouse"}), ("check_inventory", {"product_id": "mouse"})], 7, False),
    ("multi_webcam", "Locate the USB webcam, then check its stock.", [("search_product", {"query": "usb webcam"}), ("check_inventory", {"product_id": "webcam"})], 3, False),
    ("multi_stand_zero", "Search for a laptop stand and verify its inventory.", [("search_product", {"query": "laptop stand"}), ("check_inventory", {"product_id": "stand"})], 0, False),
    ("recover_webcam", "The previous inventory lookup failed. Find the USB webcam in the catalog and report its stock.", [("search_product", {"query": "usb webcam"}), ("check_inventory", {"product_id": "webcam"})], 3, True),
]


def same_arguments(actual: dict, expected: dict) -> bool:
    return actual.keys() == expected.keys() and all(
        str(actual[key]).strip().casefold() == str(value).strip().casefold()
        if key in {"query", "product_id"} and isinstance(actual[key], str) and isinstance(value, str)
        else actual[key] == value
        for key, value in expected.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/cache"))
    parser.add_argument("--output", default=str(ROOT / "outputs/metrics/rollouts.json"))
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", choices=["standard", "fresh_stress"], default="standard")
    args = parser.parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, torch_dtype="auto", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    results = []
    selected_cases = CASES if args.suite == "standard" else [(case_id, prompt, expected, stock, False) for case_id, prompt, expected, stock in fresh_stress_rollout_specs()]
    for case_id, user_prompt, expected, stock, recover in selected_cases:
        env = ProductToolEnvironment()
        messages = [{"role": "system", "content": "You are a shopping assistant. Use tools to finish the user's task. After search_product returns a matching product, call check_inventory with the returned product_id. If a tool reports an error, use the available tools to recover."}, {"role": "user", "content": user_prompt}]
        if recover:
            bad_args = {"product_id": "webcam-2024"}
            bad_result = env.execute("check_inventory", bad_args)
            messages.extend([
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "check_inventory", "arguments": bad_args}}]},
                {"role": "tool", "name": "check_inventory", "content": json.dumps(bad_result, ensure_ascii=False)},
            ])
        trace = []
        for step in range(args.max_steps):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=env.tool_definitions(), enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            text = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            action = parse_action(text)
            entry = {"step": step, "output": text, "action": action}
            trace.append(entry)
            if not action:
                break
            result = env.execute(action["name"], action["arguments"])
            entry["result"] = result
            messages.extend([
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": action["name"], "arguments": action["arguments"]}}]},
                {"role": "tool", "name": action["name"], "content": json.dumps(result, ensure_ascii=False)},
            ])
            if result.get("ok") and result.get("done", True):
                break
        actual = [(entry["action"]["name"], entry["action"]["arguments"]) for entry in trace if entry.get("action")]
        sequence_correct = len(actual) == len(expected) and all(name == exp_name and same_arguments(arguments, exp_args) for (name, arguments), (exp_name, exp_args) in zip(actual, expected))
        final_stock = trace[-1].get("result", {}).get("stock") if trace else None
        results.append({"case_id": case_id, "category": "recovery" if recover else "multi_step", "seeded_error": recover, "expected_steps": len(expected), "actual_steps": len(actual), "sequence_correct": sequence_correct, "stock_correct": final_stock == stock, "task_success": sequence_correct and final_stock == stock, "trace": trace})

    summary = {"evaluation_version": 2, "suite": "multi_step_and_recovery_rollouts" if args.suite == "standard" else "fresh_stress_rollouts", "model": args.model, "adapter": args.adapter, "device": str(model.device), "seed": args.seed, "num_cases": len(results), "task_success_rate": sum(item["task_success"] for item in results) / len(results), "multi_step": {"cases": sum(item["category"] == "multi_step" for item in results), "passed": sum(item["task_success"] for item in results if item["category"] == "multi_step")}, "recovery": {"cases": sum(item["category"] == "recovery" for item in results), "passed": sum(item["task_success"] for item in results if item["category"] == "recovery")}, "cases": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
