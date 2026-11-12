# ToolTune-Agent v2：运行、验收与面试展示

本轮修复执行和评测，没有训练新 adapter。模型为 Qwen3-0.6B；旧实验保留，当前结果以 `outputs/metrics/v2/` 和重新生成的最终报告为准。

## 六项交付

| 项目 | 实现与证据 | 边界 |
|---|---|---|
| 外部确认 | `confirm_cancellation` 是宿主接口，不暴露给模型；订单和 session 绑定，确认后一次性消费，拒绝不执行 | 本地宿主是信任边界，尚不是多用户服务身份认证 |
| 参数校验 | 工具存在、对象类型、必填、额外字段、字符串、布尔类型统一检查 | 当前支持项目四个工具所需 schema，不是完整 JSON Schema 引擎 |
| 评分修正 | 参数匹配、归一化匹配、真实环境验证分离；逐决策精确匹配同时要求工具名和参数一致 | 执行成功仍由每个案例的业务验证器判定 |
| 恢复 | 真实模型错误反馈后续调；运行中注入临时故障；测试空搜索、重试耗尽、不可恢复错误 | 故障注入不是模型自发错误；两种证据分别记录 |
| 实时推理 | `scripts/demo_model.py` 可选 Base/adapter，输出原始生成、工具反馈、完整 JSON 轨迹 | 命令行已实测；网页仍是预设轨迹，不是实时模型 UI |
| 报告复现 | 旧文件不覆盖，新评分文件保存源 SHA256；多步重新推理保存到 v2 | 浏览器视觉/HTTP 验收未完成，不宣称网页已验收 |

## 快速运行（项目根目录 PowerShell）

```powershell
$py = 'D:\Anaconda\envs\tooltune\python.exe'
$env:PYTHONPATH = 'src'
$m = 'outputs/cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
$a = 'outputs/adapters/qwen3-0.6b-pilot-500-repair'
& $py -m pytest -q
# Base：省略 adapter
& $py scripts/demo_model.py --model $m --prompt 'Find the USB webcam and then check its stock.' --output outputs/metrics/v2/my_base.json
# LoRA：现场输出来自模型，不是固定答案
& $py scripts/demo_model.py --model $m --adapter $a --prompt 'Find the USB webcam and then check its stock.' --output outputs/metrics/v2/my_repair.json
# 临时错误在首次实际库存调用时注入；其余动作由模型生成
& $py scripts/demo_model.py --model $m --adapter $a --prompt 'Find the USB webcam and then check its stock.' --test-transient-error --output outputs/metrics/v2/my_recovery.json
# 申请后暂停；只有人工输入 YES 才通过宿主授予确认
& $py scripts/demo_model.py --model $m --adapter $a --domain orders --prompt 'Please cancel order A1001.' --interactive --output outputs/metrics/v2/my_cancel.json
```

不加 `--interactive` 时，取消任务只返回 `awaiting_confirmation`，不会替用户确认。
每份输出记录模型、adapter、seed、设备、最大步数、生成长度、耗时、消息和执行轨迹。默认 seed=42、贪心解码、每次生成最多96 tokens、每次运行最多4步。暂停后的恢复是独立的一次运行，分别计步。

## 重算与重跑的区别

```powershell
# 重放保存的单步/逐决策预测，不重新生成文字
& $py scripts/rescore_v2.py
# 多步必须重新推理：新错误策略会改变后续上下文
& $py scripts/evaluate_rollouts.py --model $m --adapter $a --suite fresh_stress --output outputs/metrics/v2/fresh_stress_rollouts_repair.json
& $py scripts/build_final_report.py
```

`rescore_audit.json` 记录变化；`rescored/` 保留源路径、SHA256、评测版本。旧版多步结果只作历史记录，不能通过重放旧轨迹推算新恢复策略下的行为。

## 指标口径

- `arguments_correct`：工具名与参数严格匹配；逐决策任务也是联合匹配。
- `normalized_arguments_correct`：字符串大小写/空白归一化后的匹配，仅作诊断，不改写调用。
- `execution_success`：模型原始动作执行后，案例业务验证器通过。它不等于工具 `ok=true`，未知订单的预期错误也可通过。
- `task_success`：严格匹配且业务验证通过。
- `normalized_task_success`：归一化匹配且原始动作真实执行通过，绝不执行标准答案来替代预测。
- 多步 `task_success`：预期调用序列、参数及最终库存同时满足。命令行 `completed` 仅表示工具返回完成，不能代替多步基准成功率。
- 逐决策的历史上下文来自数据；rollout 历史来自模型自己的执行，二者不能混用。

## 面试展示顺序

1. 说明这是确定性模拟环境中的小模型工具调用研究原型。
2. 展示 Base/LoRA 的真实调用和 observation，区分模型错误、工具错误和评分失败。
3. 展示取消暂停与外部确认，解释模型参数为什么不能成为授权凭证。
4. 展示模型错误 ID 后重新搜索，以及显式标记的临时故障注入。
5. 展示最终报告的 v2 数据，说明小样本、多步顺序要求、合成模板与训练种子限制。

不要把全量规则审核称作人工标注；不要把 42 个教师历史决策点称作42个端到端任务；不要将7/14/28条的三种子试验称作500条三种子试验。Hermes 继续隔离；本轮不扩量、不新增云 GPU 支出。
