"""Live local model tool execution; no canned model outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from tooltune.agent_loop import run_agent
from tooltune.environment import OrderToolEnvironment, ProductToolEnvironment


class TransientInventoryEnvironment(ProductToolEnvironment):
    """Explicit test fault on the first actual inventory call, never a seeded message."""
    failed = False
    def execute(self,name,arguments):
        if name == "check_inventory" and not self.failed:
            self.failed = True
            result = {"ok":False,"error":"temporary_unavailable","retryable":True}
            self.trace.append({"tool":name,"arguments":arguments,"result":result})
            return result
        return super().execute(name,arguments)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--adapter")
    p.add_argument("--prompt",required=True)
    p.add_argument("--domain",choices=["orders","products"],default="products")
    p.add_argument("--max-steps",type=int,default=4)
    p.add_argument("--max-new-tokens",type=int,default=96)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--interactive",action="store_true")
    p.add_argument("--test-transient-error",action="store_true")
    p.add_argument("--output",default="outputs/metrics/v2/live_demo.json")
    args=p.parse_args()
    if args.test_transient_error and args.domain != "products":
        p.error("transient fault requires products domain")
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from tooltune.reproducibility import seed_everything
    seed_everything(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype="auto",device_map="auto")
    if args.adapter:
        model=PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    def generate(messages,tools):
        if messages and messages[-1].get("role") == "tool":
            print("TOOL_RESULT " + messages[-1]["content"], flush=True)
        prompt=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
        text=tokenizer.decode(out[0][inputs.input_ids.shape[-1]:],skip_special_tokens=True)
        print(text,flush=True)
        return text
    env=OrderToolEnvironment() if args.domain == "orders" else (TransientInventoryEnvironment() if args.test_transient_error else ProductToolEnvironment())
    messages=[{"role":"system","content":"Use tools to complete the request. For search then stock tasks use the returned product ID. Retry recoverable errors, and never invent user confirmation."},{"role":"user","content":args.prompt}]
    start=time.perf_counter()
    result=run_agent(generate,args.prompt,environment=env,messages=messages,max_steps=args.max_steps)
    runs=[result]
    if result["status"] == "awaiting_confirmation" and args.interactive:
        order=result["pending_cancellation"]
        approved=input(f"Confirm cancellation of {order}? Type YES: ") == "YES"
        env.confirm_cancellation(order,env.session_id,approved)
        if approved:
            history=result["messages"]+[{"role":"user","content":f"I confirm cancellation of order {order}."}]
            runs.append(run_agent(generate,"",environment=env,messages=history,max_steps=args.max_steps))
    report={"evaluation_version":2,"model":args.model,"adapter":args.adapter,"seed":args.seed,"device":str(model.device),"prompt":args.prompt,"domain":args.domain,"fault_injection":args.test_transient_error,"max_steps_per_run":args.max_steps,"max_new_tokens":args.max_new_tokens,"seconds":time.perf_counter()-start,"runs":runs,"environment_trace":env.trace}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),"status":runs[-1]["status"],"result":runs[-1]["result"]},ensure_ascii=True))

if __name__ == "__main__":
    main()
