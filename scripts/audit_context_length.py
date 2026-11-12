"""Audit chat-template lengths before SFT, including decision-point prefixes."""
from pathlib import Path
import argparse, json, sys
from statistics import median
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.schema import read_jsonl

def count(tokenizer, messages, tools, generation=False):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, tools=tools, enable_thinking=False)
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(ROOT/'data/processed/hermes_train_candidate_50.jsonl'))
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--output', default=None)
    p.add_argument('--model', default=str(ROOT/'outputs/cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'))
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    details=[]
    for row in read_jsonl(args.input):
        indexes=[i for i,m in enumerate(row.messages) if m.get('role')=='assistant']
        full=count(tokenizer,row.messages,row.tools)
        points=[]
        for i in indexes:
            prefix=count(tokenizer,row.messages[:i],row.tools,True)
            target=count(tokenizer,row.messages[:i+1],row.tools)-count(tokenizer,row.messages[:i],row.tools,True)
            points.append({'message_index':i,'prefix_tokens':prefix,'target_tokens':target})
        details.append({'example_id':row.example_id,'source_id':row.metadata.get('source_id'),'call_count':len(indexes),'full_tokens':full,'decision_points':points,'full_fits':full<=args.max_length,'all_decision_points_fit':all(p['prefix_tokens']+p['target_tokens']<=args.max_length for p in points)})
    values=[d['full_tokens'] for d in details]
    report={'input':str(Path(args.input)),'records':len(details),'max_length':args.max_length,'complete_under_limit':sum(d['full_fits'] for d in details),'truncated_or_over_limit':sum(not d['full_fits'] for d in details),'decision_point_records_fit':sum(d['all_decision_points_fit'] for d in details),'length_summary':{'min':min(values) if values else 0,'median':median(values) if values else 0,'p90':sorted(values)[max(0,int(len(values)*.9)-1)] if values else 0,'max':max(values) if values else 0},'lengths':details}
    output=Path(args.output) if args.output else Path(args.input).with_suffix('.context_report.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('records','max_length','complete_under_limit','truncated_or_over_limit','decision_point_records_fit','length_summary')},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
