"""Build current v2 report from rescored predictions and fresh model runs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/"outputs/metrics/v2"
def read(path): return json.loads(path.read_text(encoding="utf-8"))
def main():
    audit=read(V2/"rescore_audit.json")
    changes=[r for r in audit if any(v["old"] is not None and v["old"]!=v["new"] for v in r["changed_metrics"].values())]
    lines=["# ToolTune-Agent v2 acceptance and experiment report", "", "Validation: 33 tests passed. Local CUDA live model traces were verified; browser acceptance remains pending. Runtime versions and artifact hashes: outputs/metrics/v2/reproducibility_manifest.json.", "", "Current release: execution and scoring repair, no new training. Legacy report: `legacy_v1/final_report.md`. Chinese runbook: `../../docs/RUNBOOK_V2.md`.", "", "## Delivery status", "", "- External user confirmation: order/session-bound host event, paused agent, one-use approval and rejection tests.", "- Tool validation: required keys, types, empty strings, extra keys and unknown tools return structured errors.", "- Scoring: actual predicted actions are executed; normalization is diagnostic. Decision exactness requires both tool and arguments.", "- Recovery: runtime error feedback, empty-search retry, bounded budget and terminal-error checks; live fault-injection trace saved separately.", "- Live CLI: Base and repair adapter executed locally on CUDA. Web UI remains scripted; browser acceptance is NOT completed.", "- Reproducibility: versioned artifacts, source hashes, saved generation configuration and retained legacy outputs.", "", "## Data and training scope", "", "500 synthetic trajectories: 386 train / 76 validation / 38 entity-disjoint test. Rule/trajectory review is not independent human annotation. Repair adds 30 trajectories: 416 train trajectories / 454 decisions. Existing LoRA: rank 8, alpha 16, dropout .05, q/k/v/o, learning rate 2e-4, batch 1, accumulation 4, 200 steps, seed 42, max length 512. Three-seed experiments apply only to 7/14/28-record studies. Hermes remains quarantined.", "", "## Corrected fresh stress results", "", "Single-call: rescored saved predictions. Multi-step: newly generated with corrected error continuation. Identical suites are used within each comparison; no population-level generalization claim.", "", "| Model | Single exact | Multi-step exact |", "|---|---:|---:|"]
    for label,suffix in [("Base","base"),("Decision LoRA","decision"),("Repair LoRA","repair")]:
        single=read(V2/"rescored"/f"fresh_stress_single_{suffix}.json")
        multi=read(V2/f"fresh_stress_rollouts_{suffix}.json")
        lines.append(f"| {label} | {sum(c['task_success'] for c in single['cases'])}/{single['num_cases']} | {sum(c['task_success'] for c in multi['cases'])}/{multi['num_cases']} |")
    lines += ["", "## Scoring audit", "", f"Replayed {len(audit)} saved metric files; {len(changes)} files changed previously present metrics. Added previously missing fields are not counted as changed historical scores. Source SHA256 and provenance are embedded in each v2 result.", "", "| Source | Metric | Before | After |", "|---|---|---:|---:|"]
    for row in changes:
        for key,v in row["changed_metrics"].items():
            if v["old"] is not None:
                lines.append(f"| {row['source']} | {key} | {v['old']:.4f} | {v['new']:.4f} |")
    lines += ["", "## Live model evidence", "", "`completed` is a tool termination status, NOT proof of the requested full sequence. In particular the repair demo initially uses the wrong ID and then recovers; it is not a perfect first-pass sequence.", "", "| Artifact | Adapter | Fault injection | Terminal status | Model steps |", "|---|---|---|---|---:|"]
    for name in ["live_base.json","live_repair.json","live_repair_recovery.json","live_repair_cancel.json"]:
        r=read(V2/name);run=r["runs"][-1]
        lines.append(f"| {name} | {'repair' if r['adapter'] else 'base'} | {r['fault_injection']} | {run['status']} | {run['steps']} |")
    lines += ["", "## Important limits", "", "- Confirmation is trusted local host input, not deployed multi-user authentication.", "- Schema validation covers this project's four tools, not arbitrary JSON Schema.", "- Single-call canonical ID tasks partly test learned catalog aliases; they are not arbitrary unseen-tool generalization.", "- Historical 29/29 fixed challenge is a repair-informed regression score, not an untouched test estimate.", "- Historical 42/42 is teacher-history decision matching; seven scripted recovery checks do not measure model recovery rate.", "- Runtime fault injection and naturally occurring model mistakes are separately identified. Live smoke cases are not a recovery benchmark.", "- No new 2,000–5,000 record training, model-size scaling claim, or browser acceptance is included.", "", "## Reproduce", "", "See `docs/RUNBOOK_V2.md` for exact PowerShell commands. Run `python -m pytest -q`, `python scripts/rescore_v2.py`, the three fresh-stress rollout comparisons, and `python scripts/build_final_report.py`. Live model CLI: `scripts/demo_model.py --model PATH [--adapter PATH] --prompt TEXT`; add `--domain orders --interactive` for external confirmation."]
    out=ROOT/"outputs/reports/final_report.md"
    out.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out)
if __name__=="__main__":main()
