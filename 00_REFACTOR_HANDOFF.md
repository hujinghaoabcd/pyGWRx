# pyGWRx 重构交接入口

**任何新对话、新开发会话继续 pyGWRx 大型重构时，先读本文件。**

截至 2026-08-29，项目已经完成两轮全项目架构审计。读取顺序固定为：

1. `ARCHITECTURE_FINAL_DECISION.md` — **最终架构宪章，优先级最高**
2. `validation_results/gwr/GWR_VALIDATION_EVIDENCE.md` — 标准 GWR 数值冻结证据
3. `ARCHITECTURE_REFACTOR_MASTER_PLAN.md` — 第一轮审计历史，仅作背景；与最终决定冲突时，以 `ARCHITECTURE_FINAL_DECISION.md` 为准

继续工作时：

- 获取 `main` 当前最新 SHA；
- 读取 `ARCHITECTURE_FINAL_DECISION.md` 第 20 节执行台账；
- 只执行下一项 `PENDING` 工作；
- 每个 PR 合并后更新台账；
- 不得修改 frozen reference 数据或放宽 tolerance 来让重构通过；
- 不得让一个 public estimator 继承另一个 public estimator；
- 不得把模型数学放回 base class；
- 不得在 architecture migration PR 中顺手做性能算法重写。

当前最终设计冻结后的第一项代码工作是：

> **A1 — 19 个 estimator 的 Public API & Capability Snapshot。**
