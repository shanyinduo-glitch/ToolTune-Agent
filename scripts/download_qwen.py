"""Download Qwen3-0.6B into the project-local Hugging Face cache."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="outputs/cache")
    args = parser.parse_args()
    root = Path(args.cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.model, cache_dir=str(root))
    print(f"model={args.model}")
    print(f"snapshot={path}")


if __name__ == "__main__":
    main()
