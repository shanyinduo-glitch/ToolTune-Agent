"""Evaluate an adapter on decision points from a reviewed JSONL split."""
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
from tooltune.reproducibility import seed_everything
from tooltune.schema import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/cache"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, torch_dtype="auto", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    results = []
    for record in read_jsonl(args.data):
        for index, target_message in enumerate(record.messages):
            if target_message.get("role") != "assistant" or not target_message.get("tool_calls"):
                continue
            prefix = record.messages[:index]
            prompt = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True, tools=record.tools, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            predicted = parse_action(text)
            function = target_message["tool_calls"][0]["function"]
            expected = {"name": function["name"], "arguments": json.loads(function["arguments"])}
            results.append({"example_id": record.example_id, "decision_index": sum(1 for x in record.messages[:index] if x.get("role") == "assistant" and x.get("tool_calls")), "task_type": record.task_type, "prompt": next(m.get("content", "") for m in record.messages if m.get("role") == "user"), "expected": expected, "predicted": predicted, "parsed": predicted is not None, "tool_correct": bool(predicted and predicted.get("name") == expected["name"]), "arguments_correct": bool(predicted and predicted.get("name") == expected["name"] and predicted.get("arguments") == expected["arguments"]), "output": text})
    total = len(results)
    summary = {"model": args.model, "adapter": args.adapter, "data": args.data, "num_decisions": total,
        "parse_rate": sum(x["parsed"] for x in results) / total if total else 0,
        "tool_accuracy": sum(x["tool_correct"] for x in results) / total if total else 0,
        "argument_accuracy": sum(x["arguments_correct"] for x in results) / total if total else 0,
        "task_types": {task: {"decisions": sum(x["task_type"] == task for x in results), "exact": sum(x["task_type"] == task and x["arguments_correct"] for x in results)} for task in sorted({x["task_type"] for x in results})}, "cases": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
