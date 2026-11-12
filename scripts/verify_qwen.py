"""Load Qwen locally and run plain and tool-use prompts."""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.environment import OrderToolEnvironment


def generate(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/cache"))
    parser.add_argument("--output", default=str(ROOT / "outputs/logs/qwen3-0.6b-smoke.json"))
    args = parser.parse_args()
    model_path = args.model if Path(args.model).exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=args.cache_dir, torch_dtype="auto", device_map="auto")
    env = OrderToolEnvironment()
    plain = generate(model, tokenizer, [{"role": "user", "content": "用一句话介绍你自己。"}], 64)
    tool_messages = [{"role": "system", "content": "You are a tool-using order assistant. Return a function call when needed."}, {"role": "user", "content": "查询订单 A1001 的状态。"}]
    tool_prompt = generate(model, tokenizer, tool_messages, 96)
    result = {"model": args.model, "device": str(model.device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "plain_generation": plain, "tool_prompt_generation": tool_prompt, "tool_execution_smoke": env.execute("get_order", {"order_id": "A1001"})}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
