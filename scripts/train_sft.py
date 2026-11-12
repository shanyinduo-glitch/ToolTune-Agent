"""Minimal LoRA SFT entry point for the validated JSONL records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import read_jsonl
from tooltune.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=str(ROOT / "data/splits/train_augmented.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "outputs/adapters/qwen3-0.6b-sft"))
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--decision-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = read_jsonl(args.data)
    rows = []
    for record in records:
        assistant_indices = [i for i, message in enumerate(record.messages) if message.get("role") == "assistant"]
        if not args.decision_points:
            text = tokenizer.apply_chat_template(record.messages, tokenize=False, add_generation_prompt=False, tools=record.tools, enable_thinking=False)
            rows.append({"text": text})
            continue
        for index in assistant_indices:
            prefix = record.messages[:index]
            target_messages = record.messages[: index + 1]
            prompt = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True, tools=record.tools, enable_thinking=False)
            full = tokenizer.apply_chat_template(target_messages, tokenize=False, add_generation_prompt=False, tools=record.tools, enable_thinking=False)
            if not full.startswith(prompt):
                raise ValueError(f"chat template prefix mismatch for {record.example_id} at message {index}")
            rows.append({"prompt": prompt, "completion": full[len(prompt):]})
    dataset = Dataset.from_list(rows)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    training_args = SFTConfig(output_dir=args.output, max_steps=args.max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=1, save_strategy="no", report_to="none", fp16=False, bf16=False, remove_unused_columns=False, max_length=args.max_seq_length, completion_only_loss=args.decision_points, seed=args.seed)
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved_adapter={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
