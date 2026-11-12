"""Run a reproducible Qwen baseline over the local executable task set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.evaluation import cases, heldout_unseen_cases, unseen_cases, independent_order_cases, independent_unseen_cases, expanded_independent_order_cases, expanded_independent_unseen_cases, challenge_order_cases, challenge_unseen_cases, expanded_external_cases, fresh_stress_single_cases, evaluate_text
from tooltune.reproducibility import seed_everything


def generate(model, tokenizer, case, max_new_tokens):
    messages = [{"role": "system", "content": "You are an order assistant. Use the available tools when needed."}, {"role": "user", "content": case.user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=case.tools, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/cache"))
    parser.add_argument("--output", default=str(ROOT / "outputs/metrics/baseline.json"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--suite", choices=["orders", "heldout_orders", "unseen", "heldout_unseen", "independent_orders", "independent_unseen", "expanded_independent_orders", "expanded_independent_unseen", "challenge_orders", "challenge_unseen", "expanded_external", "fresh_stress_single"], default="orders")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, torch_dtype="auto", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
    if args.suite == "orders":
        selected_cases = cases()
    elif args.suite == "heldout_orders":
        selected_cases = [case for case in cases() if case.case_id.startswith("heldout_")]
    elif args.suite == "heldout_unseen":
        selected_cases = heldout_unseen_cases()
    elif args.suite == "independent_orders":
        selected_cases = independent_order_cases()
    elif args.suite == "independent_unseen":
        selected_cases = independent_unseen_cases()
    elif args.suite == "expanded_independent_orders":
        selected_cases = expanded_independent_order_cases()
    elif args.suite == "expanded_independent_unseen":
        selected_cases = expanded_independent_unseen_cases()
    elif args.suite == "challenge_orders":
        selected_cases = challenge_order_cases()
    elif args.suite == "challenge_unseen":
        selected_cases = challenge_unseen_cases()
    elif args.suite == "expanded_external":
        selected_cases = expanded_external_cases()
    elif args.suite == "fresh_stress_single":
        selected_cases = fresh_stress_single_cases()
    else:
        selected_cases = unseen_cases()
    results = []
    for case in selected_cases:
        text = generate(model, tokenizer, case, args.max_new_tokens)
        results.append({"case_id": case.case_id, "prompt": case.user_prompt, "output": text, **evaluate_text(text, case)})
    summary = {"evaluation_version": 2, "seed": args.seed, "model": args.model, "adapter": args.adapter, "suite": args.suite, "device": str(model.device), "num_cases": len(results), "parse_rate": sum(r["parsed"] for r in results) / len(results), "tool_accuracy": sum(r["tool_correct"] for r in results) / len(results), "argument_accuracy": sum(r["arguments_correct"] for r in results) / len(results), "normalized_argument_accuracy": sum(r["normalized_arguments_correct"] for r in results) / len(results), "normalized_task_success_rate": sum(r["normalized_task_success"] for r in results) / len(results), "task_success_rate": sum(r["task_success"] for r in results) / len(results), "failure_counts": {reason: sum(r["failure_reason"] == reason for r in results) for reason in sorted({r["failure_reason"] for r in results})}, "cases": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
