# Hermes 数据决定

**状态：暂停正式训练，继续隔离。**

当前 50 条 Hermes 候选只保留为 `review-only` / quarantine 材料，不进入训练集、验证集或留出集，也不安排云 GPU 混训。

理由：

- 按目前 512 token 的 SFT 上限，候选中只有 3/50 条完整，47/50 条超限或存在截断风险。
- 做过的解析修复只把 Python literal 格式转换为 JSON；这不是对用户请求、工具选择、参数或多步顺序的语义审核。
- 候选目前没有逐条人工审核，也没有证明它们与项目的训练集及冻结留出集相互独立。
- 之前 Hermes-only 与本地+Hermes 混训结果只作为探索性结果；不能用于声称公开数据经审核后有效或有害。

重新考虑该数据前，需先对来源与许可证逐条确认，决定并验证上下文长度处理方式，完成人工语义审核、schema/工具执行验证和污染/泄漏检查，再仅在隔离实验中比较。不得因为把样本截短或解析成功就自动放行。

依据：`data/processed/hermes_train_candidate_50.context_report.json`、`data/processed/hermes_train_candidate_50.report.json`、`data/processed/hermes_production.report.json`。
