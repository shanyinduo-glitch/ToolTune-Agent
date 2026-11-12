"""Create a conservative, quarantined Hermes production manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.schema import read_jsonl, write_jsonl

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default=str(ROOT/'data/processed/hermes_quality_approved.jsonl')); p.add_argument('--context-report',default=str(ROOT/'data/processed/hermes_train_candidate_50.context_report.json')); p.add_argument('--output',default=str(ROOT/'data/processed/hermes_production.jsonl')); p.add_argument('--report',default=str(ROOT/'data/processed/hermes_production.report.json')); p.add_argument('--max-length',type=int,default=512); a=p.parse_args()
    rows=read_jsonl(a.input); ctx={x['example_id']:x for x in json.loads(Path(a.context_report).read_text(encoding='utf-8')).get('lengths',[])}; eligible=[]; rejected=[]
    for row in rows:
        reasons=[]; flags=row.metadata.get('quality_flags',[]); item=ctx.get(row.example_id,{})
        if len([m for m in row.messages if m.get('role')=='assistant'])!=1: reasons.append('not_single_tool')
        reasons.extend(flags)
        if item.get('full_tokens',a.max_length+1)>a.max_length: reasons.append('context_over_limit')
        if reasons: rejected.append({'example_id':row.example_id,'source_id':row.metadata.get('source_id'),'reasons':sorted(set(reasons))})
        else: eligible.append(row)
    write_jsonl(a.output,[])
    report={'status':'quarantined','reason':'semantic review unavailable and context truncation risk','candidate_count':len(rows),'automatically_eligible_before_human_review':len(eligible),'eligible_count':0,'rejected_count':len(rejected),'training_allowed':False,'max_length':a.max_length,'rejected':rejected}
    Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:report[k] for k in ('status','candidate_count','automatically_eligible_before_human_review','eligible_count','training_allowed')},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
