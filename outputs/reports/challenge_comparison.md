# Challenge Set Comparison

This fixed challenge set contains paraphrases, mixed Chinese and English prompts, unknown IDs, confirmation boundaries, zero-stock products, and casing variants.

| Suite | Model | Cases | Parse | Tool | Exact args | Normalized success | Exact success | Adapter wins | Regressions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| orders | base | 12 | 75.0% | 75.0% | 41.7% | 41.7% | 41.7% | - | - |
| orders | adapter | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 7 | 0 |
| products | base | 12 | 58.3% | 58.3% | 33.3% | 50.0% | 33.3% | - | - |
| products | adapter | 12 | 100.0% | 100.0% | 83.3% | 83.3% | 83.3% | 6 | 0 |

## Multi-step and Recovery

| Scenario | Cases | Base passed | Decision LoRA passed |
|---|---:|---:|---:|
| multi step | 4 | 0 | 4 |
| recovery | 1 | 0 | 1 |

Across all 29 cases, exact task success was 31.0% for base and 93.1% for decision LoRA; parse rates on the 24 single-call cases were 66.7% and 100.0%.

The suites are fixed and deterministic. Their rates describe these cases and should not be read as population estimates. Per-case outcomes and failure categories are in `outputs/metrics/challenge_comparison.json`.
