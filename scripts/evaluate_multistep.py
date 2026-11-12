"""Evaluate an executable two-step search-to-inventory task."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/cache"))
    parser.add_argument("--output", default=str(ROOT / "outputs/metrics/multistep.json"))
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, torch_dtype="auto", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
    env = ProductToolEnvironment()
    messages = [{"role": "system", "content": "You are a shopping assistant. Complete the user's task by calling tools. After search results, use check_inventory with the returned product_id."}, {"role": "user", "content": "Find the wireless keyboard and check its inventory."}]
    expected = [("search_product", {"query": "wireless keyboard"}), ("check_inventory", {"product_id": "keyboard"})]
    trace = []
    for step in range(args.max_steps):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=env.tool_definitions(), enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        text = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        action = parse_action(text)
        item = {"step": step, "output": text, "action": action}
        if not action:
            item["result"] = None
            trace.append(item)
            break
        result = env.execute(action["name"], action["arguments"])
        item["result"] = result
        trace.append(item)
        messages.extend([{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": action["name"], "arguments": action["arguments"]}}]}, {"role": "tool", "name": action["name"], "content": json.dumps(result, ensure_ascii=False)}])
        if result.get("done", True):
            break
    actual = [(x["action"]["name"], x["action"]["arguments"]) for x in trace if x.get("action")]
    summary = {"model": args.model, "adapter": args.adapter, "task": "search_then_inventory", "steps": len(trace), "sequence_correct": actual == expected, "task_success": actual == expected and bool(trace[-1].get("result", {}).get("stock") == 12), "trace": trace}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
