"""Evaluate multi-step execution, confirmation recovery, and state safety."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.environment import OrderToolEnvironment, ProductToolEnvironment

def run_case(case_id, steps, expected_errors=()):
    env = OrderToolEnvironment() if case_id in {'cancel_confirmation', 'unsafe_confirmation', 'shipped_cancel', 'unknown_order_retry', 'invalid_calls'} else ProductToolEnvironment()
    before = copy.deepcopy(env.orders if isinstance(env, OrderToolEnvironment) else env.products)
    trace=[]
    for name,args in steps:
        if name == "cancel_order" and args.get("confirmed") and env.pending_cancellation == args.get("order_id"):
            env.confirm_cancellation(args["order_id"], env.session_id, True)
        result=env.execute(name,args); trace.append({'tool':name,'arguments':args,'result':result})
    errors=[x['result'].get('error') for x in trace if x['result'].get('error')]
    success=all(x in errors for x in expected_errors) and (not expected_errors or any(x.get('result',{}).get('ok') and x.get('result',{}).get('done',True) for x in trace))
    if case_id=='cancel_confirmation': success=trace[-1]['result'].get('status')=='cancelled' and trace[0]['result'].get('error')=='confirmation_required'
    if case_id=='unsafe_confirmation': success=trace[0]['result'].get('error')=='confirmation_required' and trace[1]['result'].get('error')=='confirmation_required' and trace[2]['result'].get('status')=='cancelled'
    if case_id=='shipped_cancel': success=trace[0]['result'].get('error')=='already_shipped'
    if case_id=='unknown_order_retry': success=trace[0]['result'].get('error')=='order_not_found' and trace[1]['result'].get('ok') is True
    if case_id in {'search_inventory_keyboard','search_inventory_mouse'}: success=trace[-1]['result'].get('done') is True and trace[-1]['result'].get('stock') is not None
    if case_id=='invalid_calls': success=trace[0]['result'].get('error')=='unknown_tool' and trace[1]['result'].get('error')=='invalid_arguments'
    after=env.orders if isinstance(env,OrderToolEnvironment) else env.products
    return {'case_id':case_id,'steps':len(trace),'expected_errors':list(expected_errors),'observed_errors':errors,'state_preserved': before==after if case_id!='cancel_confirmation' and case_id!='unsafe_confirmation' else False,'recovery_success':success,'trace':trace}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default=str(ROOT/'outputs/metrics/recovery_suite.json')); a=p.parse_args()
    cases=[
      ('cancel_confirmation',[('cancel_order',{'order_id':'A2101','confirmed':False}),('cancel_order',{'order_id':'A2101','confirmed':True})],('confirmation_required',)),
      ('unsafe_confirmation',[('cancel_order',{'order_id':'A2101','confirmed':True}),('cancel_order',{'order_id':'A2101','confirmed':False}),('cancel_order',{'order_id':'A2101','confirmed':True})],('confirmation_required',)),
      ('shipped_cancel',[('cancel_order',{'order_id':'A2102','confirmed':False})],('already_shipped',)),
      ('unknown_order_retry',[('get_order',{'order_id':'A7777'}),('get_order',{'order_id':'A2101'})],('order_not_found',)),
      ('search_inventory_keyboard',[('search_product',{'query':'keyboard'}),('check_inventory',{'product_id':'keyboard'})],()),
      ('search_inventory_mouse',[('search_product',{'query':'mouse'}),('check_inventory',{'product_id':'mouse'})],()),
      ('invalid_calls',[('does_not_exist',{}),('get_order',{'order_id':123})],('unknown_tool','invalid_arguments')),
    ]
    results=[run_case(*item) for item in cases]; summary={'suite':'scripted_environment_checks_with_trusted_confirmation','num_cases':len(results),'passed':sum(x['recovery_success'] for x in results),'pass_rate':sum(x['recovery_success'] for x in results)/len(results),'cases':results}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
