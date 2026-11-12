"""Build controlled mix manifests from the quality-gated Hermes subset."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tooltune.schema import read_jsonl, write_jsonl

def main():
    local = read_jsonl(ROOT / "data/splits/train_augmented.jsonl")
    hermes = read_jsonl(ROOT / "data/processed/hermes_quality_approved.jsonl")
    out = ROOT / "data/processed"
    write_jsonl(out / "quality_local_28.jsonl", local)
    write_jsonl(out / "quality_hermes_50.jsonl", hermes[:50])
    write_jsonl(out / "quality_local_plus_hermes_78.jsonl", local + hermes[:50])
    report = {"local": len(local), "hermes_quality_gated": min(50, len(hermes)), "local_plus_hermes": len(local) + min(50, len(hermes)), "semantic_review_flags_preserved": True}
    (out / "quality_mix.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
