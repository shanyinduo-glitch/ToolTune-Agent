# 500-Record LoRA Pilot

## Setup

- Base model: Qwen3-0.6B local snapshot (`c1899de289a04d12100db370d81485cdf75e47ca`)
- Training data: 386 entity-disjoint training trajectories from the reviewed 500-record synthetic pilot
- Validation data: 76 trajectories; 84 assistant tool decisions
- Held-out data: 38 trajectories; 42 assistant tool decisions
- Training method: decision-point completion-only LoRA, seed 42, 200 optimizer steps, batch size 1, gradient accumulation 4, learning rate 2e-4, LoRA rank 8 / alpha 16 / dropout 0.05, q/k/v/o projections
- Maximum sequence length: 512. All 414 training decision points fit; maximum rendered length was 375 tokens (95th percentile 371).
- Adapter: `outputs/adapters/qwen3-0.6b-pilot-500`
- Training runtime: 516.9 seconds; final reported training loss 0.01479.

## Results

| Evaluation | Result |
|---|---:|
| Validation decision exact match | 84/84 (100%) |
| Entity-disjoint pilot test decision exact match | 42/42 (100%) |
| Existing decision LoRA on the same pilot test | 30/42 (71.4%) |
| Base Qwen3-0.6B on the same pilot test | 14/42 (33.3%) |
| Independent order suite | 6/6 (100%) |
| Independent product suite | 5/6 (83.3%) |
| Fixed challenge: orders | 12/12 (100%) |
| Fixed challenge: products | 8/12 (66.7%) |
| Multi-step rollouts | 3/4 (75%) |
| Recovery rollout | 1/1 (100%) |
| Combined fixed challenge and rollout tasks | 24/29 (82.8%) |

The prior decision LoRA reached 27/29 (93.1%) on the same combined fixed challenge and rollout tasks. The pilot adapter improves substantially over Base and over the prior adapter on the pilot's entity-disjoint synthetic test, but it does not surpass the prior adapter on the fixed challenge suite. This is a mixed result, not an overall model improvement claim.

## Error analysis

The product challenge accounts for four failures: a Chinese product-name search produced an unusable query, a Chinese microphone request hallucinated a product ID, a laptop-stand inventory request returned natural language instead of a tool call, and uppercase query text did not exactly match the evaluator target (the last case was semantically executable but failed exact-argument scoring). The remaining multi-step failure is recorded in `outputs/metrics/pilot_500_rollouts.json`.

The 100% pilot split score should be interpreted cautiously. The data is synthetic and highly templated; train/validation/test are entity-disjoint but share prompt families and task construction. Near-zero training loss and perfect held-out entity accuracy indicate the model learned this generated format, not necessarily broad user-language generalization. The fixed external challenge results are the stronger guardrail and show remaining weaknesses.

## Artifacts

- Adapter: `outputs/adapters/qwen3-0.6b-pilot-500`
- Validation: `outputs/metrics/pilot_500_validation.json`
- Pilot test: `outputs/metrics/pilot_500_test.json`
- Base on pilot test: `outputs/metrics/base_pilot_500_test.json`
- Prior adapter on pilot test: `outputs/metrics/decision_existing_pilot_500_test.json`
- Challenge orders/products: `outputs/metrics/pilot_500_challenge_orders.json`, `outputs/metrics/pilot_500_challenge_unseen.json`
- Multi-step/recovery: `outputs/metrics/pilot_500_rollouts.json`
- Independent orders/products: `outputs/metrics/pilot_500_independent_orders.json`, `outputs/metrics/pilot_500_independent_unseen.json`
- Per-decision evaluation script: `scripts/evaluate_sft_split.py`

## Repair run and wider regression evaluation

The first pilot adapter exposed failures on Chinese product aliases and mixed-language multi-step tasks. We added 30 manually authored, execution-checked repair examples covering Chinese/traditional-Chinese product aliases, inventory wording, canonical product search terms, and search-then-inventory sequences. The repaired training file contains 416 trajectories and 454 assistant decisions. The new adapter is `outputs/adapters/qwen3-0.6b-pilot-500-repair`.

The repaired adapter again scored 42/42 exact assistant decisions on the same entity-disjoint synthetic test. This internal split remains a template-based fit check and must not be read as an estimate of real-user generalization. On the original fixed challenge and rollout suite it scored 12/12 orders, 12/12 products, 4/4 multi-step tasks, and 1/1 recovery task (29/29 combined). It scored 12/12 on a separate 12-case expanded external suite; Base scored 2/12 and the existing decision LoRA scored 9/12.

A new independently authored stress suite was added after the repair examples. It contains 14 single-call cases and four two-call search-then-inventory tasks, including Traditional Chinese aliases, ambiguous stock wording, paraphrased product searches, unknown IDs, and product-ID propagation. Results:

| Model | Fresh single-call (14) | Fresh multi-step (4) |
|---|---:|---:|
| Qwen3-0.6B Base | 4/14 (28.6%) | 0/4 |
| Existing decision LoRA | 8/14 (57.1%) | 2/4 |
| 500 repair LoRA | 9/14 (64.3%) | 2/4 |

The repaired adapter's single-call misses were three wrong product IDs/aliases, one inventory call where the request asked for a search, and one paraphrase that was not normalized to the catalog term. Its multi-step misses were a notebook-stand search that returned no catalog result and a wireless-mouse task that skipped the requested search step. The new suite therefore confirms gains over Base and the old adapter while showing that multi-step transfer is not yet stable. The per-case traces are in `outputs/metrics/fresh_stress_single_{base,decision,repair}.json` and `outputs/metrics/fresh_stress_rollouts_{base,decision,repair}.json`.

The new model comparison is deliberately kept separate from the earlier 29-case fixed challenge. The larger fixed challenge score is useful as a regression check, while the newly worded set better exposes remaining alias, intent, and tool-order errors. No overall generalization claim is made from either small suite.
