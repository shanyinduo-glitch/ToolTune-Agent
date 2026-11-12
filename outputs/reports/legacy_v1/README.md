# ToolTune-Agent

ToolTune-Agent studies tool-use capability in open language models through data preparation, supervised fine-tuning, and executable evaluation.

## Current setup

- Windows with Conda and Python 3.11
- Conda environment: `tooltune`
- Interpreter: `D:\Anaconda\envs\tooltune\python.exe`
- PyTorch: CUDA 12.6 build; verified on an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)
- Project scope: a small deterministic tool-execution harness, measured model baselines, and LoRA SFT

The existing CausaLens environments are separate and are not used by this project.

## Environment

Recreate the environment with:

```powershell
conda create -n tooltune python=3.11 -y
conda activate tooltune
python -m pip install -r requirements-cu126.txt
python -m pip install -r requirements.txt
```

Verify CUDA with:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

In PyCharm, select the existing interpreter at `D:\Anaconda\envs\tooltune\python.exe`.

## Project status

The minimal executable harness is now available:

- `src/tooltune/schema.py` defines and validates the JSONL training record format.
- `src/tooltune/environment.py` provides deterministic `get_order` and confirmation-gated `cancel_order` tools.
- `data/raw/examples.jsonl` contains a validated seed record.
- `tests/test_minimal.py` covers schema loading, tool execution, confirmation gating, and traces.

Run the checks with:

```powershell
& D:\Anaconda\envs\tooltune\python.exe -m pytest -q
```

## Challenge evaluation

The fixed challenge suite adds paraphrases, mixed Chinese and English prompts, unknown IDs, confirmation boundaries, zero-stock products, and casing variants. It is separate from the original held-out sets. Run Base and decision-LoRA evaluations and assemble their paired comparison with:

```powershell
$env:PYTHONPATH = "src"
$model = "outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
$adapter = "outputs\adapters\qwen3-0.6b-decision"
python scripts\evaluate_baseline.py --model $model --suite challenge_orders --output outputs\metrics\challenge_orders_base.json
python scripts\evaluate_baseline.py --model $model --adapter $adapter --suite challenge_orders --output outputs\metrics\challenge_orders_decision.json
python scripts\evaluate_baseline.py --model $model --suite challenge_unseen --output outputs\metrics\challenge_unseen_base.json
python scripts\evaluate_baseline.py --model $model --adapter $adapter --suite challenge_unseen --output outputs\metrics\challenge_unseen_decision.json
python scripts\compare_evaluations.py --base-orders outputs\metrics\challenge_orders_base.json --adapter-orders outputs\metrics\challenge_orders_decision.json --base-products outputs\metrics\challenge_unseen_base.json --adapter-products outputs\metrics\challenge_unseen_decision.json
```

The per-case JSON results and Markdown comparison are saved under `outputs/metrics` and `outputs/reports`. The fixed challenge set is intended for regression diagnosis and is not a population-level benchmark.

## Qwen3-0.6B baseline

The `Qwen/Qwen3-0.6B` snapshot is stored in `outputs/cache` (ignored by Git). Download or resume it with:

```powershell
& D:\Anaconda\envs\tooltune\python.exe scripts\download_qwen.py --cache-dir outputs\cache
```

Run local inference with:

```powershell
& D:\Anaconda\envs\tooltune\python.exe scripts\verify_qwen.py --model outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca --cache-dir outputs\cache
```

Observed on 2026-10-24: the model loaded successfully on `cuda:0` using the NVIDIA GeForce RTX 3060 Laptop GPU and produced output for both a plain prompt and an order-status prompt. A later baseline run supplied the native tool schema and confirmed that the model can emit parseable `<tool_call>` output; the executable evaluation below measures whether those calls are actually correct.

## Baseline evaluation and SFT

The executable baseline evaluates tool parsing, tool selection, exact arguments, and environment task success:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\evaluate_baseline.py --model outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca --cache-dir outputs\cache
```

The first local run used three deterministic cases and recorded results in `outputs/metrics/baseline.json`: parse rate 100%, tool accuracy 100%, argument accuracy 66.7%, and task success rate 66.7%. The failed case attempted cancellation with `confirmed=true` before the environment had issued a confirmation gate.

The minimal LoRA SFT entry point is `scripts/train_sft.py`. It consumes the validated JSONL format and writes an adapter under `outputs/adapters`:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\train_sft.py --model outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca --max-steps 20
```

The SFT command is intentionally a small smoke run first. It verifies tokenization, gradients, adapter creation, and checkpoint writing before expanding the dataset or moving formal training to a 24GB cloud GPU.

## Formal small-loop result

The first controlled loop used 5 training records from `data/splits/train_small.jsonl`, 40 LoRA steps, and the same 5 held-out prompt variants in the executable evaluator. The adapter is saved at `outputs/adapters/qwen3-0.6b-small-loop`.

| Model | Parse rate | Tool accuracy | Argument accuracy | Task success |
|---|---:|---:|---:|---:|
| Qwen3-0.6B base | 100% | 100% | 60% | 60% |
| Qwen3-0.6B + LoRA | 100% | 100% | 60% | 60% |

The SFT run completed with an average training loss of `0.8069` and wrote a valid adapter. The held-out behavior did not improve: both models still attempted `confirmed=true` for cancellation requests, so the environment rejected those calls. This is recorded as a negative result rather than a claimed improvement. The next data iteration should add more paraphrases and explicit positive/negative confirmation trajectories, then repeat the same fixed evaluation.

## Data and artifact policy

Keep downloaded datasets, model weights, adapters, checkpoints, logs, and caches out of Git. Review each dataset's license and access terms before use. Store large files on a drive with sufficient free space.

### 500-record synthetic review pilot

The planning target is 2,000–5,000 **quality-approved** records. The 500-record synthetic pilot contains 500 executable trajectories and 540 assistant tool decisions. It covers status lookup, confirmation-gated cancellation, inventory lookup, and two-step search/inventory trajectories. All 500 records passed full rule and trajectory review and were split by entity into 386 train, 76 validation, and 38 held-out test records. This approved synthetic pilot was used for the initial LoRA trial; its templated construction limits generalization claims. The follow-up repair adapter and external stress results are summarized in `outputs/reports/final_report.md`.

Regenerate the pilot, manifest, and stratified 30-record human-review sample with:

```powershell
& D:\Anaconda\envs\tooltune\python.exe scripts\build_synthetic_pilot.py
& D:\Anaconda\envs\tooltune\python.exe scripts\audit_training_data.py data\splits\pilot_500.jsonl
& D:\Anaconda\envs\tooltune\python.exe scripts\check_leakage.py --train data\splits\pilot_500.jsonl --eval independent
& D:\Anaconda\envs\tooltune\python.exe scripts\check_leakage.py --train data\splits\pilot_500.jsonl --eval challenge
```

The audit's near-duplicate count is a diagnostic prompt-similarity signal (template-based synthetic data naturally shares phrasing), not a standalone rejection gate. The historical review sample and manifest remain available for reproducibility. The completed full review and entity-isolated split report is `data/splits/pilot_500_reviewed/review_and_split.report.json`; audit and leakage reports are under `outputs/reports` and `outputs/metrics`.

The complete pilot review and split command is:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\review_and_split_pilot.py
```

This performs a full deterministic review of all 500 trajectories, marking the reviewed records as `semantic_review=passed` only when the prompt entity, tool, arguments, confirmation behavior, result, and multi-step chronology agree. It creates an entity-disjoint training set (`386` records), validation set (`76`), and held-out test set (`38`) under `data/splits/pilot_500_reviewed`. The split report is `review_and_split.report.json`. The held-out test set is the pilot's internal test set; keep the existing independent and challenge suites as external regression checks.

The deterministic augmentation generator is `scripts/build_small_dataset.py`. It creates 24 validated training records with paraphrases for status lookup, missing orders, confirmation-gated cancellation, and shipped-order rejection:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\build_small_dataset.py
```

The second controlled run used those 24 records for 60 steps. Its adapter is stored at `outputs/adapters/qwen3-0.6b-augmented` and its evaluation is in `outputs/metrics/sft_augmented.json`:

| Model | Parse rate | Tool accuracy | Argument accuracy | Task success |
|---|---:|---:|---:|---:|
| Base Qwen3-0.6B | 100% | 100% | 60% | 60% |
| 24-record LoRA | 100% | 100% | 100% | 100% |

The model learned the confirmation boundary on the fixed held-out set: cancellation requests now emit `confirmed=false`, and shipped-order requests are handled without attempting an unsafe cancellation. This is an initial local result on five deterministic cases, not a general benchmark claim; the next validation step should add unseen tools and larger independently generated test sets.

## Decision-point loss and multistep result

The SFT script now expands every Assistant tool decision into a separate prompt/completion record and enables completion-only loss. This prevents tool observations and later messages from being treated as the target action. The resulting 80-step adapter is `outputs/adapters/qwen3-0.6b-decision`.

On the search-then-inventory rollout, the decision-point adapter produced the complete executable sequence:

```text
search_product({"query": "wireless keyboard"})
check_inventory({"product_id": "keyboard"})
```

The multistep result is recorded in `outputs/metrics/multistep_decision.json`: sequence correctness and task success are both 100%. The same adapter also achieved 100% parse rate, argument accuracy, and task success on the five-case order suite (`outputs/metrics/decision_orders.json`). This is the first run that demonstrates a full two-tool execution loop rather than a single tool call.

## Unseen-tool generalization

The evaluator now has an `unseen` suite with `search_product` and `check_inventory`, which are absent from the SFT data. Each case records a failure reason (`parse_failure`, `wrong_tool`, `exact_argument_mismatch`, `execution_result_mismatch`, or `success`). It also reports case-folded diagnostics for string fields named `query` and `product_id`; the formal `task_success_rate` continues to require exact arguments. Run it with:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\evaluate_baseline.py --model outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca --suite unseen --output outputs\metrics\baseline_unseen.json
& D:\Anaconda\envs\tooltune\python.exe scripts\evaluate_baseline.py --model outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca --adapter outputs\adapters\qwen3-0.6b-augmented --suite unseen --output outputs\metrics\sft_augmented_unseen.json
```

Both Base and LoRA models achieved 100% parse rate, tool accuracy, argument accuracy, and task success on the three unseen-tool cases. The result shows that this Qwen checkpoint already transfers the tool schema pattern to these simple new tools; it does not prove broad tool generalization because the suite is small and deterministic.

## Expanded independent evaluation

The fixed evaluation set now contains 14 order cases and 8 product cases. The nine order cases whose IDs start with `heldout_` and the six product cases returned by `heldout_unseen_cases()` are kept separate from the training prompts:

| Model | Order held-out task success (9) | Product held-out task success (6) |
|---|---:|---:|
| Qwen3-0.6B base | 4/9 (44.4%) | 3/6 (50.0%) |
| `qwen3-0.6b-decision` | 9/9 (100%) | 5/6 (83.3%) |

The expanded full suites are recorded in `outputs/metrics/baseline_orders_expanded.json`, `decision_orders_expanded.json`, `baseline_unseen_expanded.json`, and `decision_unseen_expanded.json`. On the full suites, the base model reached 7/14 order and 5/8 product task success; the decision-point adapter reached 14/14 and 7/8. The one product failure is an argument-normalization case (`USB webcam` versus the evaluator's lower-case query), so these results measure exact executable arguments rather than semantic similarity.

## Controlled training-size study

`scripts/prepare_scale_splits.py` creates fixed-seed, task-stratified subsets from `data/splits/train_augmented.jsonl`. The formal comparison uses the same 60 training steps, LoRA configuration, seed (`42`), and held-out suites for all three sizes. The manifests in `data/splits/scales_balanced` record the exact IDs and counts.

| Training records | Split composition (confirmation / multi-step / single-tool) | Order held-out (9) | Product held-out (6) |
|---:|---:|---:|---:|
| 7 | 3 / 1 / 3 | 8/9 (88.9%) | 5/6 (83.3%) |
| 14 | 6 / 2 / 6 | 9/9 (100%) | 5/6 (83.3%) |
| 28 | 12 / 4 / 12 | 9/9 (100%) | 5/6 (83.3%) |

The seed-42 metric files are `outputs/metrics/scale_balanced_{7,14,28}_{orders,unseen}.json`. Two additional runs keep those manifests fixed and vary only training seeds 17 and 123. Across seeds 42/17/123, order task success is 8/9, 8.5/9, and 9/9 for sizes 7/14/28; product exact task success is 5/6, 5/6, and 4.33/6. The products' case-folded normalized task success is 6/6 at each size: the exact misses are query/product ID casing differences, not execution failures. Thus the observed order result improves with size, while exact product behavior does not; these small suites do not establish a general scale law. The complete per-seed results and ranges are in `outputs/metrics/scale_balanced_multiseed_summary.json`.

To reproduce the formal runs after the model snapshot is available:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\prepare_scale_splits.py --seed 42
foreach ($n in 7,14,28) {
  & D:\Anaconda\envs\tooltune\python.exe scripts\train_sft.py `
    --model outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca `
    --data "data\splits\scales_balanced\train_$n.jsonl" `
    --output "outputs\adapters\scale-balanced-$n" --max-steps 60 --seed 42
}
```

## Public function-calling data audit

Before considering external training data, the project checks license and access conditions and keeps the raw download outside Git. `Salesforce/xlam-function-calling-60k` is marked CC-BY-4.0 but gated behind the APIGen terms, so no samples were bypassed or added. The audit instead uses the openly downloadable `NousResearch/hermes-function-calling-v1` single-turn file, which is published under Apache-2.0 according to its dataset metadata.

The read-only audit command is:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\audit_hermes.py --limit 100
```

The audit examined 1,893 records. It found 1,090 records with a valid OpenAI-style tool list and parseable `<tool_call>` output: 478 single-tool records and 612 multi-step records. Among the remaining 803, ten are natural-language clarification or explanatory responses; 793 contain Python-literal payloads (for example, single-quoted strings inside otherwise JSON-like arguments), which the strict JSON parser correctly rejects. The report stores up to five raw examples per rejection class, visible tool names, and whether those names appear in the supplied schema. These records are not silently converted into training targets. The first 100 valid records are written to `data/processed/hermes_audit.jsonl`. This remains an audit artifact and is not part of any formal training split.

To separate data-subset variation from training randomness in the scale study, keep the existing seed-42 manifests fixed and run additional training seeds:

```powershell
$env:PYTHONPATH = "src"
& D:\Anaconda\envs\tooltune\python.exe scripts\run_scale_seeds.py `
  --model outputs\cache\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca `
  --seeds 17 123 --sizes 7 14 28 --max-steps 60
```

Each run writes its own adapter and held-out metrics; `outputs/metrics/scale_balanced_multiseed_summary.json` summarizes the completed seeds. The original seed-42 scale runs remain the reference runs.

For review purposes, `scripts/clean_hermes.py` creates `data/processed/hermes_literal_review.jsonl` by parsing literal payloads with `ast.literal_eval`, converting arguments back to JSON, and recording one invalid Unicode surrogate replacement. This produces 987 review candidates, but the artifact is explicitly marked `review_only` and is not a training split. Literal parsing repairs format only; candidates still require schema, semantic, and contamination review.

The next external-data experiment, if pursued, should use a separately approved subset, preserve the source IDs and license metadata, and compare against the same local held-out suites before any larger training run.

The quality audit adds a separate context-length check. With the current 512-token SFT limit, the original 50-record Hermes candidate set has only 3 records that fit completely; the other 47 are long prompts and may be truncated. Therefore the second mix run is labeled schema/format-gated, not semantically approved. The context report is `data/processed/hermes_train_candidate_50.context_report.json`.

## Controlled Hermes mix experiment

The first controlled public-data comparison selected 50 Hermes review candidates by round-robin category/subcategory from the 987 format-repaired candidates. The selected source IDs are unique and the candidate manifest is `data/processed/hermes_train_candidate_50.report.json`. Three 60-step LoRA runs used seed 42: local data only (28 records), Hermes candidates only (50), and local plus Hermes (78). The complete comparison is in `outputs/metrics/mix_summary.json`.

| Training data | Held-out orders | Held-out products | Independent orders | Independent products |
|---|---:|---:|---:|---:|
| Local 28 | 9/9 | 4/6 exact (6/6 normalized) | 6/6 | 6/6 |
| Hermes 50 | 4/9 | 3/6 exact (4/6 normalized) | 2/6 | 3/6 |
| Local + Hermes 78 | 9/9 | 4/6 exact (6/6 normalized) | 6/6 | 5/6 exact (6/6 normalized) |

In this small experiment, Hermes-only training is substantially weaker on the local executable tasks. Mixing the candidates preserves the local result and improves neither exact product accuracy nor order accuracy. This is a negative/neutral result, not evidence that the public data is generally harmful; the selected candidates were only format-repaired and were not manually approved for semantic quality. They remain experimental artifacts rather than production training data.

## Legacy multistep experiment

An earlier 28-record full-trajectory SFT run achieved 0% on the two-tool rollout. It remains in `outputs/metrics/multistep_sft.json` as a negative control. The current decision-point experiment supersedes it and is documented above.

The expanded independent suites now contain 18 order and 18 product cases. On these suites, quality-gated local training reached 18/18 and 17/18 exact; local plus Hermes reached 18/18 and 16/18 exact. Hermes-only reached 3/18 and 5/18. These figures are controlled diagnostics only because the public candidates still have semantic-review flags and context-length risk.

The expanded independent evaluation uses `--suite independent_orders` and `--suite independent_unseen` and contains 12 new prompts. `scripts/check_leakage.py` compares these prompts with training prompts using exact matching and a 0.92 normalized similarity threshold; the current report has zero exact or near matches.

## Final report, recovery evaluation, and demo

The final report is generated by `scripts/build_final_report.py` and stored at `outputs/reports/final_report.md`. `scripts/plot_results.py` generates the three PNG figures under `outputs/figures`. `scripts/evaluate_recovery.py` runs seven deterministic multi-step and error-recovery cases and writes `outputs/metrics/recovery_suite.json`.

Hermes candidates are quarantined in `data/processed/hermes_production.jsonl` because automatic schema checks are not human semantic review and 47/50 candidates exceed the 512-token SFT context limit. Existing Hermes mix adapters remain exploratory artifacts only.

The Streamlit prototype is `app.py`. In the current Conda environment Streamlit is not installed; use the zero-dependency browser workbench with `D:\Anaconda\envs\tooltune\python.exe demo_server.py`. It serves deterministic task traces, current model comparisons, and selected failures at `http://localhost:8765`.


## Current 500-record follow-up status

The 500-record pilot now has a repair LoRA adapter (`outputs/adapters/qwen3-0.6b-pilot-500-repair`) and an independently authored 18-case fresh stress set. See `outputs/reports/final_report.md` for the measured results and `outputs/reports/hermes_data_decision.md` for the Hermes quarantine decision. The fresh stress score is 9/14 single-call and 2/4 multi-step for the repair adapter; larger data expansion is deferred until alias mapping and multi-step reliability improve.

Start the zero-dependency browser workbench from the project root with:

```powershell
& D:\Anaconda\envs\tooltune\python.exe demo_server.py
```

Open `http://localhost:8765`. The workbench includes deterministic order lookup, confirmation-gated cancellation, product search and inventory, an error-recovery trace, recorded model comparisons, and failed stress examples. It does not run live model inference. `scripts/build_final_report.py` assembles the Markdown report from metric artifacts, and `scripts/plot_results.py` regenerates the PNG figures.
