# pyGWRx 重构交接入口

**任何新对话、新开发会话继续 pyGWRx 大型重构时，先读本文件。**

截至 2026-08-29，项目已经完成两轮全项目架构审计，并已经进入最终架构执行阶段。读取顺序固定为：

1. `ARCHITECTURE_FINAL_DECISION.md` — **最终架构宪章，优先级最高**
2. `architecture_contracts/estimators.json` — A1：19 个 public estimator 的 API / capability 冻结
3. `architecture_contracts/FITTED_STATE_ATOMICITY.md` — A2：19 模型 failed-refit 原子性合同
4. `architecture_contracts/migration_risks.json` 与 `architecture_contracts/MIGRATION_RISK_MATRIX.md` — A3：19 模型迁移风险矩阵
5. `validation_results/gwr/GWR_VALIDATION_EVIDENCE.md` — 标准 GWR 数值冻结证据
6. `FUTURE_MODEL_DEVELOPMENT_PLAN.md` — 0.2 架构稳定后的未来模型扩展路线，不属于当前重构范围
7. `ARCHITECTURE_REFACTOR_MASTER_PLAN.md` — 第一轮审计历史，仅作背景；与最终决定冲突时，以 `ARCHITECTURE_FINAL_DECISION.md` 为准

继续工作时：

- 获取 `main` 当前最新 SHA；
- 读取 `ARCHITECTURE_FINAL_DECISION.md` 第 20 节执行台账；
- 结合本文件的“当前状态”判断实际下一项工作；
- 只执行下一项尚未完成的工作；
- 每个 PR 合并后更新执行状态/台账；
- 不得修改 frozen reference 数据或放宽 tolerance 来让重构通过；
- 不得让一个 public estimator 继承另一个 public estimator；
- 不得把模型数学放回 base class；
- 不得在 architecture migration PR 中顺手做性能算法重写；
- 不得提前实现 `FUTURE_MODEL_DEVELOPMENT_PLAN.md` 中的未来算法，除非 0.2 架构执行台账已经允许进入未来模型开发阶段。

## 强制跨窗口接力规则

每完成一个阶段或一个可独立交接的 PR，当前对话必须在给用户的回复末尾额外提供一段 **“给新窗口的接力说明”**。这不是可选项。

接力说明至少必须包含：

1. 当前 `main` SHA；
2. 最近完成并合并的阶段 / PR / merge SHA；
3. 当前正在进行的阶段和分支；
4. 已冻结、不可随意改变的合同或数值 gate；
5. 下一项应该执行的具体工作；
6. 当前已知风险、禁止事项或未解决阻塞；
7. 提醒新窗口先读本文件和 `ARCHITECTURE_FINAL_DECISION.md`。

目的：即使原对话达到上下文上限，新窗口也能仅凭仓库文件 + 这段接力说明继续工作，而不需要重新做架构判断。

## 当前状态

- Z1 — 最终架构裁决：**DONE**，PR #31，merge SHA `a7374d65296adc77a8a390ea9e97c76e8116352f`。
- A1 — Public API & capability snapshot：**DONE**，PR #34，merge SHA `888f9ceac8fd2c988afb4a004056d5c523302290`。
- A2 — 19-model fitted-state atomicity freeze：**DONE**，PR #35，merge SHA `b655688f7201aaa9677fe153f2cbc15e6e63afb6`。
- A3 — 19-model migration risk matrix：**DONE**，PR #36，merge SHA `dc150ef09fed370a45e5a6e62f14846979fff643`。
- A4 — performance/memory baseline harness：**IN PROGRESS**，分支 `refactor/a4-performance-memory-baseline`。
- A4 完成后下一项固定为：**A5 — pre-refactor contract/message hygiene**。

A1/A2/A3 共同构成后续迁移的安全冻结：Public API / capability、failed-refit atomicity、迁移风险与执行约束必须同时守住；标准 GWR 的独立 reference evidence 继续作为 blocking numerical gate。A4 只建立可复现的性能/内存 baseline 与结构型 memory regression guard，不改变 estimator 数学、执行策略或统计公式，也不设置 noisy wall-time CI gate。
