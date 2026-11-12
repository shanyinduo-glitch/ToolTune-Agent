"""Check exact and near-duplicate user prompts across train and evaluation data."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import read_jsonl


def norm(value: str) -> str:
    return " ".join(value.casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data/splits/train_augmented.jsonl"))
    parser.add_argument("--eval", default="independent")
    parser.add_argument("--threshold", type=float, default=0.92)
    args = parser.parse_args()
    train = read_jsonl(args.train)
    if args.eval == "independent":
        from tooltune.evaluation import expanded_independent_order_cases, expanded_independent_unseen_cases
        eval_rows = expanded_independent_order_cases() + expanded_independent_unseen_cases()
        eval_prompts = [(x.case_id, x.user_prompt) for x in eval_rows]
    elif args.eval == "challenge":
        from tooltune.evaluation import challenge_order_cases, challenge_unseen_cases
        eval_rows = challenge_order_cases() + challenge_unseen_cases()
        eval_prompts = [(x.case_id, x.user_prompt) for x in eval_rows]
    else:
        eval_rows = read_jsonl(args.eval)
        eval_prompts = [(x.example_id, next(m["content"] for m in x.messages if m.get("role") == "user")) for x in eval_rows]
    train_prompts = [(x.example_id, next(m["content"] for m in x.messages if m.get("role") == "user")) for x in train]
    exact, near = [], []
    for eval_id, prompt in eval_prompts:
        ep = norm(prompt)
        for train_id, train_prompt in train_prompts:
            score = SequenceMatcher(None, ep, norm(train_prompt)).ratio()
            if score == 1.0:
                exact.append({"eval_id": eval_id, "train_id": train_id, "score": score})
            elif score >= args.threshold:
                near.append({"eval_id": eval_id, "train_id": train_id, "score": round(score, 4)})
    report = {"train_file": args.train, "evaluation_set": args.eval, "train_records": len(train), "eval_records": len(eval_prompts), "threshold": args.threshold, "exact_matches": exact, "near_matches": near, "leakage_free": not exact and not near}
    suffix = "pilot_500_" if "pilot_500" in Path(args.train).name else ""
    out = ROOT / "outputs/metrics" / f"{suffix}leakage_{args.eval}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if exact or near else 0)


if __name__ == "__main__":
    main()
