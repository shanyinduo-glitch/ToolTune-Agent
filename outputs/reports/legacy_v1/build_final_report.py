"""Assemble the final reproducibility report from recorded artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def ratio(metric: dict, key: str = "task_success_rate") -> str:
    return f"{metric[key] * 100:.1f}% ({sum(case['task_success'] for case in metric['cases'])}/{metric['num_cases']})"


def main() -> None:
    independent_orders = read("outputs/metrics/independent_orders_decision.json")
    independent_products = read("outputs/metrics/independent_unseen_decision.json")
    scale = read("outputs/metrics/scale_balanced_multiseed_summary.json")
    recovery = read("outputs/metrics/recovery_suite.json")
    hermes_context = read("data/processed/hermes_train_candidate_50.context_report.json")
    hermes_quarantine = read("data/processed/hermes_production.report.json")
    stress = {
        "Base": (read("outputs/metrics/fresh_stress_single_base.json"), read("outputs/metrics/fresh_stress_rollouts_base.json")),
        "Decision LoRA": (read("outputs/metrics/fresh_stress_single_decision.json"), read("outputs/metrics/fresh_stress_rollouts_decision.json")),
        "500 repair LoRA": (read("outputs/metrics/fresh_stress_single_repair.json"), read("outputs/metrics/fresh_stress_rollouts_repair.json")),
    }

    external = {
        "Base": read("outputs/metrics/base_expanded_external.json"),
        "Decision LoRA": read("outputs/metrics/decision_expanded_external.json"),
        "500 repair LoRA": read("outputs/metrics/pilot_500_repair_expanded_external.json"),
    }
    pilot_repair_test = read("outputs/metrics/pilot_500_repair_test.json")
    challenge_orders = read("outputs/metrics/pilot_500_repair_challenge_orders.json")
    challenge_products = read("outputs/metrics/pilot_500_repair_challenge_unseen.json")
    challenge_rollouts = read("outputs/metrics/pilot_500_repair_rollouts.json")
    entities = read("data/splits/pilot_500_reviewed/review_and_split.report.json")

    lines = [
        "# ToolTune-Agent 最终实验报告",
        "",
        "## 项目目标与结论",
        "",
        "本项目使用可执行的订单与商品工具环境，评估 Qwen3-0.6B 的工具选择、参数生成、多步调用和错误恢复，并通过 LoRA 训练闭合数据、训练、执行评测和演示流程。",
        "",
        f"- 既有 decision LoRA 在独立订单集为 {ratio(independent_orders)}，独立商品集为 {ratio(independent_products)}。",
        f"- 七项恢复套件通过 {recovery['passed']}/{recovery['num_cases']}。",
        f"- 三随机种子规模试验中订单成功率均值由 7 条数据时 {scale['sizes']['7']['orders']['mean_task_success'] * 100:.1f}% 增至 28 条时 {scale['sizes']['28']['orders']['mean_task_success'] * 100:.1f}%；样本与任务集仍不足以支持普遍规模规律。",
        "- 本轮新增 18 个独立压力任务后，500 条修复 LoRA 的单步成功率为 9/14，多步成功率为 2/4。小样本结果显示明显收益，也显示中文商品别名、搜索/查库存意图区分和产品映射仍是瓶颈。",
        "- 因压力集上的多步表现尚未稳定，当前不启动 2,000–5,000 条扩量；应先扩展经人工核验的多步/别名数据，并在冻结留出集上复测。",
        "",
        "## 500 条试点与修复训练",
        "",
        f"全量审核后的 500 条合成轨迹按实体隔离为训练 {entities['split_records']['train']}、验证 {entities['split_records']['validation']}、测试 {entities['split_records']['test']} 条。首次试点与后续修复 adapter 在实体隔离测试上均为 {pilot_repair_test['num_decisions']}/{pilot_repair_test['num_decisions']} 决策精确匹配；这反映合成模板内插值，不代表广泛泛化。",
        "",
        f"修复 adapter 的固定挑战订单 {ratio(challenge_orders)}、商品 {ratio(challenge_products)}；旧的五条多步/恢复任务通过 {challenge_rollouts['multi_step']['passed']}/{challenge_rollouts['multi_step']['cases']} 多步与 {challenge_rollouts['recovery']['passed']}/{challenge_rollouts['recovery']['cases']} 恢复。独立编写的 12 条 expanded external 案例结果为 Base {ratio(external['Base'])}、旧 decision LoRA {ratio(external['Decision LoRA'])}、500 repair LoRA {ratio(external['500 repair LoRA'])}。",
        "",
        "## 新增独立压力评测",
        "",
        "压力集独立于修复训练提示，包含 14 个单步案例和 4 个两工具任务，涉及订单边界、未知 ID、繁体中文商品别名、零库存、自然语言改写以及 search 后根据返回 ID 查库存。每个指标都由对应确定性工具环境执行验证。",
        "",
        "| 模型 | 单步成功 | 多步成功 |",
        "|---|---:|---:|",
    ]
    for label, (single, rollouts) in stress.items():
        lines.append(f"| {label} | {ratio(single)} | {rollouts['multi_step']['passed']}/{rollouts['multi_step']['cases']} |")

    lines.extend([
        "",
        "修复 LoRA 的 5 个单步失败包括 3 个错误商品 ID/别名映射、1 个将搜索请求错作库存查询、1 个对同义表达未映射到目录关键词。多步中的笔记本支架搜索未返回规范商品词；无线鼠标案例则直接查库存，没有按请求先搜索。Base 的四个多步任务均未完成；旧 decision LoRA 完成 2/4。所有逐条输出与失败轨迹见 `outputs/metrics/fresh_stress_single_*.json` 和 `outputs/metrics/fresh_stress_rollouts_*.json`。",
        "",
        "该新增集是小型诊断套件，不是总体用户分布估计。它揭示了原固定挑战集接近饱和与新表达泛化之间的差距，因此本轮 29/29 固定挑战成绩不能单独支撑扩量决策。",
        "",
        "## Hermes 数据决定",
        "",
        f"暂停 Hermes 正式训练，保留候选为隔离的 review-only 数据。当前 512 token 上限下只有 {hermes_context['complete_under_limit']}/{hermes_context['records']} 条完整，{hermes_context['truncated_or_over_limit']} 条超限或有截断风险；50 条候选没有完成逐条人工语义审核，隔离清单许可为 `training_allowed={hermes_quarantine['training_allowed']}`。当前不投入云 GPU 处理 Hermes。重新评估前须完成来源/许可证确认、上下文处理、语义审核和训练/留出污染检查。决策记录见 `outputs/reports/hermes_data_decision.md`。",
        "",
        "## 图表、评测与演示",
        "",
        "图表位于 `outputs/figures/`：独立任务成功率、三种子规模实验、Hermes 上下文门槛及新增压力集对比。浏览器演示使用 `python demo_server.py` 启动，展示确定性订单查询、取消确认、商品多步查询、错误恢复和已记录的模型评测摘要；它不进行实时模型推理。Streamlit 入口是 `app.py`，但当前 Conda 环境未安装 Streamlit，因此已验证零依赖演示服务作为可运行版本。",
        "",
        "## 可复现产物",
        "",
        "- 500 条数据审核与切分：`data/splits/pilot_500_reviewed/review_and_split.report.json`",
        "- repair adapter 训练与逐决策评测：`outputs/adapters/qwen3-0.6b-pilot-500-repair`、`outputs/metrics/pilot_500_repair_test.json`",
        "- 独立压力单步/rollout：`outputs/metrics/fresh_stress_single_*.json`、`outputs/metrics/fresh_stress_rollouts_*.json`",
        "- Hermes 决策：`outputs/reports/hermes_data_decision.md`",
        "- 浏览器演示：`demo_server.py`；Streamlit 演示：`app.py`",
        "",
        "## 局限",
        "",
        "工具环境仅包含少量确定性订单和商品；大部分训练数据为合成数据，提示分布和工具 schema 固定。成功率受精确参数判分约束，尤其商品同义词必须映射到目录内规范词。独立压力集也只有 18 项，适合定位故障，不构成稳健统计估计。尚未完成更大数据训练、跨工具/API 泛化、多人盲审一致性和实时模型演示。",
        "",
    ])
    output = ROOT / "outputs/reports/final_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
