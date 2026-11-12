# ToolTune-Agent v2 acceptance and experiment report

Validation: 33 tests passed. Local CUDA live model traces were verified; browser acceptance remains pending. Runtime versions and artifact hashes: outputs/metrics/v2/reproducibility_manifest.json.

Current release: execution and scoring repair, no new training. Legacy report: `legacy_v1/final_report.md`. Chinese runbook: `../../docs/RUNBOOK_V2.md`.

## Delivery status

- External user confirmation: order/session-bound host event, paused agent, one-use approval and rejection tests.
- Tool validation: required keys, types, empty strings, extra keys and unknown tools return structured errors.
- Scoring: actual predicted actions are executed; normalization is diagnostic. Decision exactness requires both tool and arguments.
- Recovery: runtime error feedback, empty-search retry, bounded budget and terminal-error checks; live fault-injection trace saved separately.
- Live CLI: Base and repair adapter executed locally on CUDA. Web UI remains scripted; browser acceptance is NOT completed.
- Reproducibility: versioned artifacts, source hashes, saved generation configuration and retained legacy outputs.

## Data and training scope

500 synthetic trajectories: 386 train / 76 validation / 38 entity-disjoint test. Rule/trajectory review is not independent human annotation. Repair adds 30 trajectories: 416 train trajectories / 454 decisions. Existing LoRA: rank 8, alpha 16, dropout .05, q/k/v/o, learning rate 2e-4, batch 1, accumulation 4, 200 steps, seed 42, max length 512. Three-seed experiments apply only to 7/14/28-record studies. Hermes remains quarantined.

## Corrected fresh stress results

Single-call: rescored saved predictions. Multi-step: newly generated with corrected error continuation. Identical suites are used within each comparison; no population-level generalization claim.

| Model | Single exact | Multi-step exact |
|---|---:|---:|
| Base | 4/14 | 0/4 |
| Decision LoRA | 8/14 | 2/4 |
| Repair LoRA | 9/14 | 2/4 |

## Scoring audit

Replayed 82 saved metric files; 6 files changed previously present metrics. Added previously missing fields are not counted as changed historical scores. Source SHA256 and provenance are embedded in each v2 result.

| Source | Metric | Before | After |
|---|---|---:|---:|
| mix_local-28_heldout_unseen.json | normalized_task_success_rate | 1.0000 | 0.8333 |
| mix_local-plus-hermes-78_heldout_unseen.json | normalized_task_success_rate | 1.0000 | 0.8333 |
| quality_local-28_expanded_independent_unseen.json | normalized_task_success_rate | 1.0000 | 0.9444 |
| quality_local-plus-hermes-78_expanded_independent_unseen.json | normalized_task_success_rate | 1.0000 | 0.9444 |
| scale_balanced_28_trainseed_123_unseen.json | normalized_task_success_rate | 1.0000 | 0.8333 |
| scale_balanced_28_trainseed_17_unseen.json | normalized_task_success_rate | 1.0000 | 0.8333 |

## Live model evidence

`completed` is a tool termination status, NOT proof of the requested full sequence. In particular the repair demo initially uses the wrong ID and then recovers; it is not a perfect first-pass sequence.

| Artifact | Adapter | Fault injection | Terminal status | Model steps |
|---|---|---|---|---:|
| live_base.json | base | False | completed | 3 |
| live_repair.json | repair | False | completed | 3 |
| live_repair_recovery.json | repair | True | completed | 3 |
| live_repair_cancel.json | repair | False | awaiting_confirmation | 1 |

## Important limits

- Confirmation is trusted local host input, not deployed multi-user authentication.
- Schema validation covers this project's four tools, not arbitrary JSON Schema.
- Single-call canonical ID tasks partly test learned catalog aliases; they are not arbitrary unseen-tool generalization.
- Historical 29/29 fixed challenge is a repair-informed regression score, not an untouched test estimate.
- Historical 42/42 is teacher-history decision matching; seven scripted recovery checks do not measure model recovery rate.
- Runtime fault injection and naturally occurring model mistakes are separately identified. Live smoke cases are not a recovery benchmark.
- No new 2,000–5,000 record training, model-size scaling claim, or browser acceptance is included.

## Reproduce

See `docs/RUNBOOK_V2.md` for exact PowerShell commands. Run `python -m pytest -q`, `python scripts/rescore_v2.py`, the three fresh-stress rollout comparisons, and `python scripts/build_final_report.py`. Live model CLI: `scripts/demo_model.py --model PATH [--adapter PATH] --prompt TEXT`; add `--domain orders --interactive` for external confirmation.
