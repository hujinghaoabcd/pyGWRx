# pyGWRx 重构交接入口

**任何新对话、新开发会话继续 pyGWRx 大型重构时，先读本文件。**

截至 2026-08-30，项目已经完成两轮全项目架构审计，并已经进入最终架构执行阶段。读取顺序固定为：

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
- A4 — performance/memory baseline harness：**DONE**，PR #38，merge SHA `de64389163fc5094ab46b72671872d26f35b9733`。
- A5 — pre-refactor contract/message hygiene：**DONE**，PR #40，merge SHA `595a632076025eb9b45794caaff37e3b3747a700`。GTWR 正权重样本不足时的误导性 `ridge regularized` warning 已改为真实的 rank-aware minimum-norm unpenalized WLS 描述，并用回归测试同时锁定 warning 文案与 minimum-norm 行为；未改变 solver、公式或估计结果。
- B1 — private `_protocols.py`：**DONE**，PR #42，merge SHA `039bceee97a4b62518b7efc686350f3cb1beeaf2`。新增 fitted lifecycle、regression surface、parameter inference、temporal view、multiscale view、weight provider、stored weight components 七类 private structural Protocol；加入 mypy gate，并用架构测试锁定七类能力且确认未从 `pygwrx.core` 或包根公开导出；未修改 estimator、base、solver、统计公式或执行策略。
- B2 — validation split：**DONE**，PR #44，merge SHA `e95e616fc11e2a315890efe9af21262756ec9491`。新增 canonical `pygwrx.core.validation`，把 model-independent coordinate/data validation 从 `utils.py` 独立出来；`pygwrx.core.utils.validate_coords` / `validate_data` 继续作为 compatibility re-export，旧公共导入保持有效；新模块加入 blocking mypy spine，并以 contract tests 锁定 canonical ownership 与兼容 identity；未修改 estimator、base、distance/kernel/bandwidth/solver 数学、执行策略、reference fixture 或 tolerance。
- B3 — distance split + metric spec：**DONE**，PR #46，merge SHA `24edf6902f1ae180ab29d6b29ce6680b205b9625`。新增 canonical `pygwrx.core.distance`，集中 ordinary metric、`DistanceCache` 与 bounded row/block streaming；`pygwrx.core.utils` 保留 identical compatibility aliases，既有 `pygwrx.core` public distance names/signatures 不变；新增 private-first `DistanceMetricSpec` 与 ordinary metric params 支持，并明确排除 GTWR/STWR/LGGWR/GRGWR/ScalableGWR 等 model-specific geometry；通过 blocking mypy、generated-doc、完整平台矩阵、coverage、Security/SBOM 与独立 GWR reference gates；未修改 estimator、base、kernel/bandwidth neighbourhood 语义、solver、统计公式、reference fixture 或 tolerance。
- B4 — time-axis utility：**DONE**，PR #48，merge SHA `90cd65bc8342b82538afa9faabb32eb41a58d979`。新增 canonical private-first `pygwrx.core.time` 与 `TimeAxis`，统一 numeric/datetime time-axis normalization、resolved unit/origin 与 prediction-time compatibility；GTWR 已改用该 canonical utility，同时保留 `time_unit_`、`time_origin_`、`time_input_kind_`、`times_train_` 等冻结 fitted-state contract；未修改 GTWR/MGTWR/STWR/SGTWR 的时空距离或 temporal-effect 数学，SGTWR 的 public-estimator-as-utility 兼容路径仍按 D5 处理。
- B5 — weights + neighbourhood policies：**DONE**，PR #50，merge SHA `cdb2bfed50a107d1bed2b19380eff7ab8dea1763`。新增 private-first `pygwrx.core.weights`，建立 `FixedBandwidth`、`AdaptiveBandwidth` 与 `NeighbourhoodPolicy`，显式冻结 focal-counting、k-th boundary、duplicate-zero fallback、tie handling 与 LOOCV focal-exclusion 语义；分别锁定 GWR/MGWR 的 distance-threshold inclusive / `nextafter` 规则，以及 GWPCA/GWDA/GWSS 的 stable-rank / exact-kernel-boundary 规则，并用合同测试直接与现有生产实现对照；尚未把 estimator 执行路径迁移到新模块，未修改 solver、bandwidth search、kernel、统计公式、reference fixture 或 tolerance。
- B6 — canonical WLS result：**DONE**，PR #52，merge SHA `318caa5b17416ae2fa01ccef5f4142a35db76713`。将现有 WLS details 正规化为 canonical private frozen result 与 `_solve_weighted_least_squares` 纯代数入口；正式系数字段为 `params`，保留 `.beta` private compatibility property 与 `_weighted_least_squares_details` exact alias，因此当前 GWR 生产代码无需迁移；公开 `weighted_least_squares(...)->(beta, inverse_normal)` 签名和二元组返回保持不变。SVD、rank cutoff、ridge、minimum-norm 与 inverse-normal 数值语义均未改变，solver contracts、独立 GWR references、完整 CI/Docs/Security 全部通过。
- B7 — generic search domains：**DONE**，PR #54，merge SHA `467aa2b21b23109d4a1c64b99a63fb0538b48431`。在 `pygwrx.core.optimization` 建立 private frozen continuous/integer search-domain value objects，并让 canonical Golden-section 内部路径显式按 domain 分派；公开 `GoldenSectionSearch.minimize(..., adaptive=False)` 继续作为 0.1.x compatibility wrapper。GTWR 已改为显式传递 integer/continuous domain，objective 仍由 GTWR 模型拥有；Golden continuous/integer、Brent、bandwidth selector 的算法、候选顺序、tie/caching/tolerance/rounding 与统计定义均未改变，contract tests、独立 numerical references、完整 CI/Docs/Security 全部通过。
- B8 — Gaussian diagnostics split：**DONE**，PR #56，merge SHA `d8ec00378b01ad1d9b1d3692f6c149bf3f4637ae`。新增 canonical `pygwrx.core.gaussian_diagnostics`，Gaussian AIC/AICc/BIC、local R²、trace(S)/trace(S'S)、ENP/EDF 与 `compute_diagnostics` 迁入 canonical owner；`pygwrx.core.metrics` 保留 distribution-neutral R²，并对既有 Gaussian diagnostic 名称提供 exact compatibility re-export，`pygwrx.core` public names 不变。split/statistical contracts、独立 GWR numerical references、coverage、minimum dependencies、build/install、完整平台矩阵、Documentation 与 Security/SBOM 全部通过；未改变诊断公式、AIC/AICc/BIC/ENP/EDF 定义、public result 语义、模型数值、reference fixture 或 tolerance。
- C1 — GWR private engine extraction：**DONE**，PR #59，merge SHA `69d2f441a29cf02976bc1b1502193da70c2b23e8`。新增 private `pygwrx.models._gwr_engine`，把 standard GWR 的 bandwidth objective、spatial-weight orchestration、Gaussian local WLS orchestration、prediction-location fitting、inference collection 与 rank-policy hook 从 public estimator shell 中提取；`GWR` public constructor/method/result/fitted-state contract 保持不变，现有受 RGWR 等依赖的 protected methods 仅保留为 thin compatibility wrappers，未提前执行 D1 concrete-inheritance cleanup。private-vs-public selector trace contracts、fitted-state atomicity、independent numerical references、coverage、minimum dependencies、build/install、Ubuntu/macOS/Windows × Python 3.11–3.14、Documentation 与 Security/SBOM 全部通过；未修改统计公式、search candidate/tie semantics、reference fixture 或 tolerance。
- 下一项固定为：**C2 — GWR engine numerical lock**。

A1/A2/A3/A4/A5、B1/B2/B3/B4/B5/B6/B7/B8 与 C1 共同构成进入 C2 的安全冻结：Public API / capability、failed-refit atomicity、迁移风险、性能/内存现状、用户可观察契约、private capability spine、canonical validation/distance/time/weights/WLS/search-domain/Gaussian-diagnostics 边界，以及 standard-GWR private engine ownership 必须同时守住；标准 GWR 的独立 reference evidence 继续作为 blocking numerical gate。A4 的 timing/RSS 仅作为 observational baseline，不设置 noisy wall-time CI gate；live structural memory guards 继续阻止 streamed GWR 意外引入 retained `n × n` buffer，并保留 `compute_hat_matrix=True` 的显式 dense S 行为。C2 只允许补强并冻结 GWR engine 的 numerical/behavior gates（external references、deep hat/influence/Cook's D、rank-deficient behavior、prediction、streaming、bandwidth provenance、failed-refit atomicity），不得放宽 tolerance、修改 reference fixture、改变 estimator API/统计公式/search semantics，不得提前进入 D1 或后续模型迁移、base rewrite、concrete inheritance 移除或性能算法重写。
