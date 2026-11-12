"""Rescore saved predictions without altering legacy artifacts or generating new text."""
import json
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tooltune import evaluation as ev
from tooltune.actions import parse_action


def main():
    lookup={c.case_id:c for fn in (ev.cases,ev.unseen_cases,ev.independent_order_cases,ev.independent_unseen_cases,ev.expanded_independent_order_cases,ev.expanded_independent_unseen_cases,ev.challenge_order_cases,ev.challenge_unseen_cases,ev.expanded_external_cases,ev.fresh_stress_single_cases) for c in fn()}
    outdir=ROOT/"outputs/metrics/v2/rescored";outdir.mkdir(parents=True,exist_ok=True)
    audit=[]
    for path in sorted((ROOT/"outputs/metrics").glob("*.json")):
        old=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(old,dict):
            continue
        cases=old.get("cases")
        if not isinstance(cases,list) or not cases or not all(isinstance(c,dict) and "output" in c for c in cases):
            continue
        new=dict(old)
        if "num_decisions" in old and all("expected" in c for c in cases):
            rows=[]
            for c in cases:
                r=dict(c);pred=parse_action(c["output"]);expected=c["expected"]
                r.update(predicted=pred,parsed=pred is not None,tool_correct=bool(pred and pred["name"]==expected["name"]),arguments_correct=bool(pred and pred==expected))
                rows.append(r)
            new["task_types"]={t:{"decisions":sum(c["task_type"]==t for c in rows),"exact":sum(c["task_type"]==t and c["arguments_correct"] for c in rows)} for t in {c["task_type"] for c in rows}}
        elif all(c.get("case_id") in lookup for c in cases):
            rows=[{**c,**ev.evaluate_text(c["output"],lookup[c["case_id"]])} for c in cases]
            for key in ("task_success","normalized_task_success","execution_success"):
                new[key+"_rate"]=sum(c[key] for c in rows)/len(rows)
            new["normalized_argument_accuracy"]=sum(c["normalized_arguments_correct"] for c in rows)/len(rows)
            new["failure_counts"]={r:sum(c["failure_reason"]==r for c in rows) for r in {c["failure_reason"] for c in rows}}
        else:
            continue
        for field,key in (("parse_rate","parsed"),("tool_accuracy","tool_correct"),("argument_accuracy","arguments_correct")):
            new[field]=sum(c[key] for c in rows)/len(rows)
        new.update(cases=rows,evaluation_version=2,source_file=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method="replay_saved_predictions_no_new_inference")
        (outdir/path.name).write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding="utf-8")
        changes={k:{"old":old.get(k),"new":new[k]} for k in ("task_success_rate","normalized_task_success_rate","argument_accuracy") if k in new and old.get(k)!=new[k]}
        audit.append({"source":path.name,"cases":len(rows),"changed_metrics":changes})
    (outdir.parent/"rescore_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"files":len(audit),"changed":[x for x in audit if x["changed_metrics"]]},indent=2))
if __name__=="__main__":main()
