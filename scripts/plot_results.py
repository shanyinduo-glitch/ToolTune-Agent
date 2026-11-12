"""Build compact PNG figures from recorded metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)


def metric(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str = "Task success") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=["#2f6f9f", "#78a6c8", "#d9822b", "#4c956c"][: len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    save_bar(
        OUT / "task_success_comparison.png",
        ["base\norders", "decision\norders", "base\nproducts", "decision\nproducts"],
        [4 / 6, 6 / 6, 3 / 6, 6 / 6],
        "Independent held-out task success",
    )
    report = metric("outputs/metrics/scale_balanced_multiseed_summary.json")
    sizes = ["7", "14", "28"]
    means = [report["sizes"][size]["orders"]["mean_task_success"] for size in sizes]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sizes, means, marker="o", color="#2f6f9f")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training records")
    ax.set_ylabel("Mean order task success")
    ax.set_title("Scale study across three training seeds")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "scale_multiseed.png", dpi=160)
    plt.close(fig)
    audit = metric("data/processed/hermes_train_candidate_50.context_report.json")
    save_bar(
        OUT / "hermes_audit.png",
        ["complete\n<=512", "over\n512"],
        [audit["complete_under_limit"] / audit["records"], audit["truncated_or_over_limit"] / audit["records"]],
        "Hermes context-length gate",
        ylabel="Share of candidates",
    )
    single = [metric(f"outputs/metrics/fresh_stress_single_{name}.json")["task_success_rate"] for name in ("base", "decision", "repair")]
    rollout = [metric(f"outputs/metrics/fresh_stress_rollouts_{name}.json")["task_success_rate"] for name in ("base", "decision", "repair")]
    save_bar(
        OUT / "fresh_stress_comparison.png",
        ["base\nsingle", "decision\nsingle", "repair\nsingle", "base\nmulti", "decision\nmulti", "repair\nmulti"],
        single + rollout,
        "Fresh stress regression",
    )
    print("\n".join(str(path.resolve()) for path in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
