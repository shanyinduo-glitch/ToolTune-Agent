"""Zero-dependency browser demo fallback when Streamlit is unavailable."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from tooltune.environment import OrderToolEnvironment, ProductToolEnvironment

HTML = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ToolTune-Agent Workbench</title>
<style>
:root{font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif;color:#202522;background:#f6f7f6}*{box-sizing:border-box}body{margin:0}header{background:#182c35;color:#fff;padding:22px max(20px,calc((100vw - 1080px)/2))}header h1{font-size:24px;margin:0 0 4px}header p{margin:0;color:#d5e0e3}main{max-width:1080px;margin:auto;padding:20px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}button{font:inherit;cursor:pointer;border:1px solid #bec8c5;background:#fff;color:#202522;padding:9px 12px;border-radius:4px}button[aria-pressed=true]{background:#176b61;color:#fff;border-color:#176b61}button.primary{background:#176b61;color:white;border-color:#176b61}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:16px}select{font:inherit;padding:9px;border:1px solid #bec8c5;border-radius:4px;background:white}.summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:16px 0}.metric{background:#fff;border-top:3px solid #176b61;padding:12px}.metric strong{display:block;font-size:20px}.metric span{color:#4b5753}.panel{background:white;padding:14px;margin:16px 0;border:1px solid #dce1df;border-radius:4px}.trace{display:grid;gap:8px}.step{border-left:3px solid #288579;background:#f3f7f5;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:7px 0 0;font:13px/1.5 ui-monospace,Consolas,monospace}.error{border-color:#b54737}.subtle{color:#52605c;font-size:13px}a{color:#176b61} @media(max-width:620px){main{padding:14px}.summary{grid-template-columns:1fr}.row{align-items:stretch}select{width:100%}button{min-height:42px}}
</style>
<header><h1>ToolTune-Agent</h1><p>模型工具调用评测与确定性执行工作台</p></header>
<main><div class="tabs" role="tablist"><button data-view="demo" aria-pressed="true">工具演示</button><button data-view="metrics" aria-pressed="false">实验结果</button><button data-view="failure" aria-pressed="false">失败分析</button></div>
<section id="view-demo"><div class="row"><select id="task" aria-label="选择演示任务"><option value="order">查询订单状态</option><option value="cancel">确认取消订单</option><option value="product">搜索商品并查库存</option><option value="recovery">错误后重新搜索并查库存</option></select><button class="primary" id="run">执行</button></div><div id="trace" class="panel subtle">选择任务后执行，查看真实环境调用顺序和返回值。</div></section>
<section id="view-metrics" hidden></section><section id="view-failure" hidden></section></main>
<script>
const buttons=[...document.querySelectorAll('[data-view]')];buttons.forEach(b=>b.onclick=()=>{buttons.forEach(x=>x.setAttribute('aria-pressed',String(x===b)));document.querySelectorAll('main section[id^="view-"]').forEach(s=>s.hidden=s.id!==`view-${b.dataset.view}`)});
document.getElementById('run').onclick=async()=>{const out=document.getElementById('trace');out.textContent='执行中…';try{const data=await (await fetch('/api/run?task='+document.getElementById('task').value)).json();out.className='panel';out.innerHTML='<h2>调用轨迹</h2><div class="trace">'+data.trace.map((s,i)=>`<article class="step"><strong>步骤 ${i+1} · ${s.tool}</strong><pre>${escapeHtml(JSON.stringify({arguments:s.arguments,result:s.result},null,2))}</pre></article>`).join('')+'</div><h3>任务状态</h3><pre>'+escapeHtml(JSON.stringify(data.final_state,null,2))+'</pre>'}catch(e){out.className='panel error';out.textContent='演示请求失败：'+e}};
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function load(){const r=await fetch('/api/metrics');const m=await r.json();document.getElementById('view-metrics').innerHTML='<div class="summary">'+m.cards.map(x=>`<div class="metric"><span>${x.label}</span><strong>${x.value}</strong></div>`).join('')+'</div><div class="panel"><h2>新增独立压力集</h2><table><thead><tr><th>模型</th><th>单步（14）</th><th>多步（4）</th></tr></thead><tbody>'+m.stress.map(x=>`<tr><td>${x.name}</td><td>${x.single}</td><td>${x.multi}</td></tr>`).join('')+'</tbody></table><p class="subtle">小样本诊断集，仅用于比较和定位错误。</p></div><p><a href="/report">查看最终实验报告</a></p>';
document.getElementById('view-failure').innerHTML='<div class="panel"><h2>压力集失败点</h2><ul>'+m.failures.map(x=>`<li>${escapeHtml(x)}</li>`).join('')+'</ul><h3>真实失败样例</h3>'+m.examples.map(x=>`<article class="step"><strong>${escapeHtml(x.case_id)} · ${escapeHtml(x.failure_reason||'未完成两步链路')}</strong><p>${escapeHtml(x.prompt||'')}</p><pre>${escapeHtml(x.output||JSON.stringify(x.trace,null,2))}</pre></article>`).join('')+'<h3>数据决策</h3><p>Hermes 候选暂缓训练；新压力集上的多步成功仍为 2/4，暂不启动 2,000–5,000 条扩量。</p></div>'}
load();
</script></html>"""


def json_artifact(path: str) -> dict:
    artifact = ROOT / path
    return json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else {}


def metric_payload() -> dict:
    pairs = [("Base", "base"), ("Decision LoRA", "decision"), ("500 repair LoRA", "repair")]
    stress = []
    for label, suffix in pairs:
        single = json_artifact(f"outputs/metrics/fresh_stress_single_{suffix}.json")
        multi = json_artifact(f"outputs/metrics/fresh_stress_rollouts_{suffix}.json")
        if single and multi:
            stress.append({"name": label, "single": f"{sum(x['task_success'] for x in single['cases'])}/{single['num_cases']}", "multi": f"{multi['multi_step']['passed']}/{multi['multi_step']['cases']}"})
    challenge = json_artifact("outputs/metrics/pilot_500_repair_challenge_orders.json")
    product = json_artifact("outputs/metrics/pilot_500_repair_challenge_unseen.json")
    external = json_artifact("outputs/metrics/pilot_500_repair_expanded_external.json")
    rollout = json_artifact("outputs/metrics/pilot_500_repair_rollouts.json")
    summary = json_artifact("outputs/metrics/pilot_500_repair_test.json")
    def fraction(obj: dict, field: str = "cases") -> str:
        return f"{sum(c['task_success'] for c in obj.get(field, []))}/{obj.get('num_cases', 0)}" if obj else "未运行"
    cards = [
        {"label": "实体隔离试点决策", "value": f"{summary.get('num_decisions', '—')}/{summary.get('num_decisions', '—')}" if summary else "未运行"},
        {"label": "固定挑战集 24 项", "value": f"{fraction(challenge)} + {fraction(product)}"},
        {"label": "多步 / 错误恢复", "value": f"{rollout.get('multi_step', {}).get('passed', '—')}/4 · {rollout.get('recovery', {}).get('passed', '—')}/1"},
    ]
    failure_lines = [
        "Repair LoRA 新单步压力集：9/14；主要失败为商品别名/规范 ID 映射、搜索与库存意图区分。",
        "Repair LoRA 新多步压力集：2/4；商品搜索未命中或跳过先搜索步骤。",
        f"扩展独立集：{fraction(external)}；固定挑战成绩不能替代新措辞压力测试。",
    ]
    single_repair = json_artifact("outputs/metrics/fresh_stress_single_repair.json")
    multi_repair = json_artifact("outputs/metrics/fresh_stress_rollouts_repair.json")
    examples = [
        {"case_id": item["case_id"], "prompt": item["prompt"], "output": item["output"], "failure_reason": item["failure_reason"]}
        for item in single_repair.get("cases", []) if not item.get("task_success")
    ][:3]
    examples.extend(
        {"case_id": item["case_id"], "trace": item["trace"]}
        for item in multi_repair.get("cases", []) if not item.get("task_success")
    )
    return {"cards": cards, "stress": stress, "failures": failure_lines, "examples": examples}


class Handler(BaseHTTPRequestHandler):
    def respond(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            self.respond(json.dumps(metric_payload(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return
        if parsed.path == "/report":
            report = ROOT / "outputs/reports/final_report.md"
            content = report.read_text(encoding="utf-8") if report.exists() else "报告尚未生成。"
            self.respond(("<!doctype html><meta charset=utf-8><pre style='white-space:pre-wrap;font:15px/1.6 system-ui;max-width:900px;margin:32px auto'>" + content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>").encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/run":
            task = parse_qs(parsed.query).get("task", [""])[0]
            if task == "order":
                env = OrderToolEnvironment()
                steps = [("get_order", {"order_id": "A2101"})]
            elif task == "cancel":
                env = OrderToolEnvironment()
                steps = [("cancel_order", {"order_id": "A2101", "confirmed": False}), ("cancel_order", {"order_id": "A2101", "confirmed": True})]
            elif task == "product":
                env = ProductToolEnvironment()
                steps = [("search_product", {"query": "wireless keyboard"}), ("check_inventory", {"product_id": "keyboard"})]
            elif task == "recovery":
                env = ProductToolEnvironment()
                steps = [("check_inventory", {"product_id": "webcam-2024"}), ("search_product", {"query": "usb webcam"}), ("check_inventory", {"product_id": "webcam"})]
            else:
                self.send_error(404, "Unknown demo task")
                return
            trace = []
            for name, args in steps:
                if name == "cancel_order" and args.get("confirmed"):
                    # Fixed fixture, explicitly labelled: never a live model authorization.
                    granted = env.confirm_cancellation(args["order_id"], env.session_id, True)
                    trace.append({"tool": "scripted_user_confirmation_event", "arguments": {"order_id": args["order_id"]}, "result": {"authorized": granted, "fixture_only": True}})
                trace.append({"tool": name, "arguments": args, "result": env.execute(name, args)})
            final_state = env.orders if isinstance(env, OrderToolEnvironment) else env.products
            body = json.dumps({"task": task, "trace": trace, "final_state": final_state}, ensure_ascii=False).encode()
            self.respond(body, "application/json; charset=utf-8")
            return
        self.respond(HTML.encode("utf-8"), "text/html; charset=utf-8")

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
