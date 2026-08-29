# pyGWRx 最终架构设计决定

> **状态：FINAL / ARCHITECTURE FREEZE**  
> **日期：2026-08-29**  
> **代码审计基线：`main` @ `da506c6f37b6154aa16105042464bc8960f8f036`**  
> **文档分支基线：`main` @ `fb41ab45dbda7f96f4f6e31e9832a6e69457106c`（与上述审计基线运行时代码相同）**  
> **适用范围：pyGWRx 0.1.x → 0.2.0 大型架构重构**
>
> 本文件是 pyGWRx 下一阶段重构的**最终架构决定（architecture decision record + execution constitution）**。
> 它在架构目标、模块边界、继承策略、执行顺序和验收规则上**取代** `ARCHITECTURE_REFACTOR_MASTER_PLAN.md`。
> 旧 master plan 继续保留，作为第一次全项目审计的证据和历史记录；若两者冲突，以本文件为准。
>
> 本文件的目的不是“提出更多可能方案”，而是**停止继续讨论架构方向，冻结最终方案，然后分阶段实现**。

---

# 0. 给任何新对话 / 新开发者 / Codex 会话的第一条指令

开始任何 pyGWRx 架构修改前，严格按以下顺序：

1. 读取根目录 `00_REFACTOR_HANDOFF.md`。
2. 完整读取本文件 `ARCHITECTURE_FINAL_DECISION.md`。
3. 读取 `validation_results/gwr/GWR_VALIDATION_EVIDENCE.md`。
4. 获取 GitHub `main` **当前最新 SHA**，不得假定仍是本文件审计基线。
5. 检查本文件第 20 节“最终执行台账”。
6. 只做下一项 `PENDING` 工作；每个 PR 只完成一个明确架构任务。
7. 开 PR 前重新搜索受影响符号的全部生产代码依赖，不能凭本文件中的旧行号直接修改。
8. PR 合并前必须通过对应层级的 numerical / contract / CI / docs / security gate。
9. PR 合并后更新执行台账，记录 PR、merge SHA、数值验证和架构验证。
10. **禁止为了让重构通过而修改 frozen reference 数据、放宽数值 tolerance、删除失败测试或改变统计定义。**

如果代码现状与本文件冲突：

- 先判断是否是本文件之后出现的新需求；
- 若只是实现偏离，恢复到本文件定义的架构；
- 只有出现新的、可证明的数学/软件约束时，才能通过新的 ADR 明确修改本文件中的冻结决定；
- 不允许在普通模型 PR 中顺手改变架构原则。

---

# 1. 为什么需要第二次独立深审计

第一次审计正确识别了“半统一架构”：base、core、models、diagnostics、plotting 之间存在大量重复和隐式耦合，并提出了“继承负责契约、组合负责数值能力”的方向。

第二次审计刻意从三个互相独立的角度重新验证：

1. **统计/数值线**：如果统一 core，会不会把统计含义不同的模型错误统一？
2. **软件架构线**：当前继承树、结果协议和 fitted-state 是否能稳定支撑全部 19 个 estimator？
3. **性能/内存线**：如果统一距离/权重后端，会不会让 MGWR/MGTWR/ScalableGWR 等模型出现灾难性的时间或内存回退？

交叉裁决后发现：第一次方案方向正确，但还不够彻底。如果直接按旧 B1–H4 开工，仍有几处很可能在中后期再次推翻：

- root base 仍然太重；
- `BaseSpatialRegressor.fit(X, y, coords)` 式 ABC 契约并不适合 STWR/GTWR/MGTWR 等异构输入；
- 把所有数据类型塞进 `contracts.py` 会形成新的 junk drawer；
- 一个 `DistanceSpec` 无法同时表达普通距离、时空距离、latent geometry 和 kNN 压缩算法；
- “统一 adaptive bandwidth 规则”会冒着改变 GWmodel-compatible 模型边界语义的风险；
- “stream everything”会严重伤害 MGWR/MGTWR 的重复 backfitting；
- concrete estimator inheritance（例如 `GWGLM(GWR)`、`MGTWR(MGWR)`）仍会持续传播父类隐式状态；
- 直接统一结果类会把完全不同的统计结果塞进一个巨型 optional dataclass；
- 在 architecture refactor 前缺少全部 estimator 的公开接口与行为 characterization freeze。

因此，本文件对第一版路线作出最终修正。

---

# 2. 最终架构的十条不可违反原则

## F1. 一个极薄 root；算法不进入 base

`BaseSpatialEstimator` 只负责生命周期和最小元数据机制。

它**不得拥有**：

- `distance_metric`
- kernel
- bandwidth
- adaptive/fixed 语义
- bandwidth selector
- solver
- local regression
- prediction interpolation/recalibration
- Gaussian diagnostics
- `to_frame()` 的系数假设
- 时空距离
- multiscale 状态

## F2. public estimator 禁止继承另一个 public estimator

最终代码中不允许：

```python
class RGWR(GWR): ...
class LCRGWR(GWR): ...
class GWGLM(GWR): ...
class MGTWR(MGWR): ...
```

**任何 public estimator 都不能把另一个 public estimator 当实现基类。**

如果一个模型需要 GWR/MGWR/GTWR 的算法能力，必须依赖 private engine / capability component。

## F3. 继承只表示角色；组合表示算法

最终 public estimator 只继承薄角色基类，例如：

- `BaseSpatialRegressor`
- `BaseSpatialClassifier`
- `BaseSpatialTransformer`
- `BaseSpatialStatistics`
- `BaseSpatialInference`

这些角色基类是 state-light / algorithm-free。

## F4. 不再建立深的 spatiotemporal / multiscale implementation hierarchy

`BaseSpatiotemporalRegressor`、`BaseMultiscaleRegressor` 不再承担实现能力。

0.1.x 可保留为兼容层；0.2.0 中要么成为无状态 marker/alias，要么弃用。

时间、多尺度是**capability**，不是一棵算法继承树。

## F5. core 只收“模型无关的数学/基础设施原语”

若一个算法只有某个模型族使用，优先放在 `models/_..._engine.py`，而不是为了“统一”塞进 core。

## F6. 数学语义与执行策略完全分离

例如“Gaussian kernel + adaptive k”是数学语义；“距离 streaming / dense cache / kNN”是执行策略。

不得为了性能优化改变统计定义，也不得为了统一统计定义强迫所有模型使用同一内存策略。

## F7. 共享基础设施可以参数化语义，但不能抹平模型差异

特别是：

- adaptive k 是否包含 focal observation；
- compact kernel 的第 k 个边界点是否通过 `nextafter` 纳入；
- duplicate coordinates 如何处理；
- LOOCV 时 focal weight 如何置零；
- ridge 是否存在、是否惩罚 intercept；
- AIC/AICc 的复杂度定义；

必须由明确的 model/family policy 决定。

## F8. public result 保持 model-specific；下游通过 private view/protocol 统一

不创建“万能 PredictionResult / ModelResult”。

GWR、GTWR、GWGLM、LGGWR 等继续拥有自己的 public result dataclass。

## F9. 新架构内部对象默认 private

新的 Protocol、engine、execution policy、internal result 等一律先以 `_` 私有模块/名称存在。

在 0.2.0 之前不因为“看起来通用”就加入 `pygwrx.core.__all__`。

## F10. 先冻结行为，再改结构

高风险模型必须先有 characterization/reference gate，再允许迁移架构。

---

# 3. 最终顶层对象模型

## 3.1 最终 inheritance 结构

目标不是复杂的 class taxonomy，而是非常浅的角色结构：

```text
BaseSpatialEstimator
├── BaseSpatialRegressor
├── BaseSpatialClassifier
├── BaseSpatialTransformer
├── BaseSpatialStatistics
└── BaseSpatialInference
```

这些类全部满足：

- 不定义某一种空间权重；
- 不定义 fit 的固定参数签名；
- 不定义某一种 predict 签名；
- 不定义 bandwidth；
- 不定义 kernel；
- 不定义 solver；
- 不假定训练坐标就是唯一结果坐标；
- 不假定结果一定存在 `coef_`。

### 为什么不继续使用严格 ABC `fit(X, y, coords)`

因为 19 个模型的自然接口不同：

- GWR：`fit(X, y, coords)`
- GTWR/MGTWR：多一个 `times`
- STWR：多阶段 `X_list / y_list / coords_list / time intervals`
- GWPCA：没有 `y`
- GWDA：分类接口
- GWSS：统计接口
- BootstrapGWR：推断而不是 prediction estimator

强迫它们共享一个 abstract method signature 只会导致 `**kwargs`、错误类型声明和下一轮 base 重构。

### root base 最终职责

`BaseSpatialEstimator` 最多保留：

```text
_is_fitted
is_fitted_
_mark_fitted()
_mark_unfitted()
_check_is_fitted()
```

以及**真正无语义争议**的元数据小工具，例如：

```text
_capture_feature_names(...)
_clear_common_metadata()
```

是否存储 `n_samples_ / n_features_in_ / feature_names_in_` 由具体 estimator 在 fit 时决定；root 不强迫所有模型具有完全相同的数据结构。

### random_state / verbose

它们也不是 root 必需配置。需要它们的 estimator 自己声明并拥有。

---

# 4. 19 个 estimator 的最终角色与实现归属

| Public estimator | 最终角色 base | 禁止的 concrete parent | 主要 private engine / capability | 目标执行策略 |
|---|---|---|---|---|
| GWR | BaseSpatialRegressor | — | Gaussian local regression + GWR bandwidth objective | block stream 默认；hat 可选 dense |
| RGWR | BaseSpatialRegressor | GWR | GWR local engine + robust outer reweighting | stream/cache 由 GWR engine |
| LCRGWR | BaseSpatialRegressor | GWR | LCR solver + GWR spatial weight capability | block stream 可行；LCR 数值专用 |
| MGWR | BaseSpatialRegressor | heavy multiscale base | multiscale backfitter + exact smoother engine | repeated-use dense/cache/auto |
| GTWR | BaseSpatialRegressor | heavy ST base | GTWR geometry + Gaussian local engine | cached ST components / later auto policy |
| MGTWR | BaseSpatialRegressor | MGWR | MGTWR scale search + multiscale ST backfitter | cache spatial/time base；combined 按需 |
| GWGLM | BaseSpatialRegressor | GWR | family-specific IWLS engine | spatial weights provider + IWLS |
| GWLasso | BaseSpatialRegressor | — | local standardized L1 solver + local CV | model-specific |
| MixedGWR | BaseSpatialRegressor | — | `_mixed_gwr_core` partial regression | model-specific |
| ScalableGWR | BaseSpatialRegressor | — | ScaGWR compressed-moment/kNN engine | kNN + compressed moments，绝不普通 dense GWR |
| SGWR | BaseSpatialRegressor | — | spatial + similarity weight providers + local Gaussian engine | on-demand weights；dense only if requested |
| SGTWR | BaseSpatialRegressor | — | ST + similarity providers + local engine | on-demand / cache policy |
| STWR | BaseSpatialRegressor | — | STWR stage engine | staged/history-specific |
| LGGWR | BaseSpatialRegressor | — | latent geometry optimizer + local Gaussian final fit | learned geometry；model-specific |
| GRGWR | BaseSpatialRegressor | — | regime discovery + regime local smoother | graph/sparse + local model |
| GWPCA | BaseSpatialTransformer | — | local weighted PCA engine | dense/blocks according to dataset |
| GWDA | BaseSpatialClassifier | — | local discriminant statistics engine | dense/blocks according to dataset |
| GWSS | BaseSpatialStatistics | — | local weighted statistics engine | dense/blocks according to dataset |
| BootstrapGWR | BaseSpatialInference | — | bootstrap runner + GWR engine/factory | replicate-level orchestration |

**冻结决定：**不因为 MGWR/GTWR/MGTWR 有特殊 capability 就建立新的深继承树；capability 用 Protocol / component 表达。

---

# 5. 最终目录边界

目标内部结构如下。具体 private 文件可以按模型规模稍作拆分，但**职责边界不可反转**。

```text
src/pygwrx/
├── core/
│   ├── base.py                 # 极薄生命周期 / role bases
│   ├── _protocols.py           # private structural protocols only
│   ├── validation.py           # 通用数组/feature/coordinate validation
│   ├── distance.py             # 普通数值距离 + streaming/cache primitives
│   ├── time.py                 # numeric/datetime time-axis normalization
│   ├── kernels.py              # pure K(d / h)
│   ├── weights.py              # explicit bandwidth/neighbourhood semantics
│   ├── optimization.py         # generic scalar/discrete search only
│   ├── solver.py               # pure WLS linear algebra primitives
│   ├── metrics.py              # distribution-neutral metrics
│   ├── gaussian_diagnostics.py # smoother/Gaussian regression diagnostics
│   ├── _execution.py           # private execution/memory policies
│   └── _summary.py
│
├── models/
│   ├── gwr.py                  # public estimator/result shell
│   ├── _gwr_engine.py          # GWR-specific objective/orchestration
│   ├── rgwr.py
│   ├── _rgwr_engine.py
│   ├── lcr_gwr.py
│   ├── _lcr_engine.py
│   ├── mgwr.py
│   ├── _mgwr_engine.py
│   ├── gtwr.py
│   ├── _gtwr_geometry.py
│   ├── _gtwr_engine.py
│   ├── mgtwr.py
│   ├── _mgtwr_engine.py
│   ├── glm_gwr.py
│   ├── _gwglm_engine.py
│   ├── ...
│   └── model-specific private engines
│
├── diagnostics/
│   ├── ...                     # consume private capability views/protocols
│
├── plotting/
│   ├── ...                     # consume diagnostics/views; no model math
│
└── io/
    └── ...
```

## 5.1 不创建新的 `core/contracts.py`

旧计划中的 `contracts.py` 被取消。

理由：把 DistanceSpec、LocalSolveResult、SearchResult、Protocol 全部放在一个模块，会迅速形成第二个 `utils.py`。

最终规则是 **type lives with its owner**：

- solver result → `solver.py`
- optimization result/domain → `optimization.py`
- metric spec → `distance.py`
- Protocol → `_protocols.py`
- model-specific backfit result → model private engine

---

# 6. Distance 最终设计：metric 与 geometry 分层

## 6.1 普通数值 metric

`core.distance` 负责：

- Euclidean
- Manhattan/cityblock
- Chebyshev
- Minkowski
- Haversine
- pairwise matrix
- bounded blocks/rows

内部规范对象建议固定为：

```python
@dataclass(frozen=True)
class DistanceMetricSpec:
    name: str
    params: Mapping[str, float]
```

它解决当前 API 中 `minkowski` 没有 `p`、Haversine radius 无法从 estimator 传入的问题。

0.1.x 保持现有 `distance_metric` 兼容；0.2 API 若扩展参数，统一使用 `metric_params`，不要每种 metric 新增一个 constructor 参数。

## 6.2 model-specific geometry 不进入 DistanceMetricSpec

下列东西**不是**普通 metric string：

- GTWR `lambda/ksi/tau/causal` 时空组合
- MGTWR coefficient-specific tau
- STWR response-variation temporal effect
- LGGWR latent geometry
- GRGWR graph/regime geometry
- ScalableGWR kNN/compressed neighbourhood

它们必须由 model private geometry/provider 实现。

## 6.3 DistanceProvider 是 capability，不是基类

可以定义 private Protocol，但不让 estimator 继承它。

其任务是生成模型定义下的距离 row/block；具体缓存方式由 execution policy 决定。

---

# 7. Time 最终设计

当前 SGTWR 用一个 `GTWR` 实例承担时间转换，这是错误的依赖方向：完整 estimator 被当成 utility。

最终引入 `core.time`：

- numeric time normalization
- datetime → numeric conversion
- resolved time unit
- prediction time compatibility checks

建议内部值对象：

```python
@dataclass(frozen=True)
class TimeAxis:
    values: np.ndarray
    unit: str
    origin: object | None
    datetime_like: bool
```

GTWR、SGTWR、MGTWR 等复用 TimeAxis；**时空距离公式仍由模型自己实现**。

STWR 的 response-variation temporal effect不能被强行转成 GTWR distance。

---

# 8. Weight / bandwidth 语义最终设计

这是第二次审计对旧方案最重要的修正之一。

## 8.1 不存在一个无条件“统一 adaptive bandwidth 规则”

当前模型中存在不同参考实现语义：

- 标准 GWR 为 compact kernel 使用 `nextafter(kth_distance, +inf)` 纳入第 k 边界邻居；
- MGWR 具有 vectorized adaptive weight path 和 reference-specific discrete search；
- GWPCA/GWDA/GWSS 使用 stable rank / GWmodel-compatible 规则；
- duplicate coordinates 的 fallback 也并不完全相同。

因此 core 必须共享**机制**，同时显式携带**语义 policy**。

## 8.2 内部 bandwidth 类型

公共 API 继续接受 number + `adaptive`，但进入 engine 后马上规范化为内部类型：

```text
FixedBandwidth(value)
AdaptiveBandwidth(k, neighbourhood_policy)
```

不要求用户直接构造这些类型。

## 8.3 NeighbourhoodPolicy 必须显式定义

至少描述：

- calibration 时 focal observation 是否计入 k；
- compact kernel 第 k 个边界是否强制包含；
- zero-distance duplicates 的 fallback；
- stable tie ordering；
- LOOCV focal exclusion发生在哪一层。

**禁止**在未来 PR 中因为“去重代码”把一个模型的 neighbourhood policy 替换成另一个模型的规则，除非外部数值验证证明完全等价。

## 8.4 WeightProvider 是一等 capability

模型的数学权重应由 provider 生成，而不是 diagnostics/plotting 重新推导。

private `WeightProviderProtocol` 至少支持按 target row/block 获取模型一致权重。

权重矩阵是否存储是另一个问题：

```text
weight calculation != weight storage
```

`store_weights=True` 只是结果保留策略，不是算法必须 materialize dense matrix 的理由。

---

# 9. Solver 最终设计

## 9.1 core WLS 是纯线性代数

最终内部 canonical function 只接受：

```text
X, y, weights, explicit numerical options
```

它不知道：

- coords
- distance
- kernel
- bandwidth
- target location
- GWR
- GTWR

内部 result 与 solver 同模块，例如：

```text
WeightedLeastSquaresResult
- params
- inverse_normal / generalized inverse information
- rank
- singular_values (if useful)
- condition_number
```

现有 public `weighted_least_squares()` tuple API 在兼容期作为 wrapper 保留。

## 9.2 绝不建立“一个 solver 统治所有模型”

下列必须保留专用数学：

- LCRGWR：local compensated ridge + 特定 condition-number 约定
- GWLasso：local standardisation + coordinate descent / L1
- GWGLM：IWLS / family-specific working weights
- MixedGWR：partial regression
- ScalableGWR：compressed cross-products + global shrinkage
- GWPCA：SVD
- GWDA：local covariance/discriminant

它们可以复用低层线性代数 helper，但不能为了代码统一改变估计量。

## 9.3 rank deficiency policy 属于 estimator/family

WLS core 报告 rank；由 estimator 决定：

- 是否允许 minimum-norm coefficient；
- inference 是否 NaN；
- 是否 raise；
- 是否允许 explicit ridge。

标准 GWR 已冻结：rank-deficient location 保留 minimum-norm coefficient/prediction，但 coefficient inference 为 NaN。

---

# 10. Gaussian local fitting engine

为了避免 GWR、GTWR、RGWR、SGWR 等重复“遍历 target → 权重 → WLS → hat row → trace”，允许建立一个**private Gaussian local fitting primitive/engine**。

它必须满足：

- 输入是已经定义好的 weight rows/provider；
- solver 通过明确 dependency 注入；
- 不计算距离；
- 不决定 bandwidth；
- 不决定 robust residual weights；
- 不决定 time metric；
- 不决定 similarity formula；
- 不决定 model-specific rank policy。

因此它是 local-linear execution engine，不是新的“BaseGWR”。

**禁止再引入任何名为 `BaseGWR` 的实现基类。**

---

# 11. Search / bandwidth selection 最终设计

## 11.1 generic optimizer 不知道 adaptive

`GoldenSectionSearch.minimize(... adaptive=True)` 这种把统计语义塞进 optimizer 的设计最终应消失。

Generic search 只理解 domain：

```text
ContinuousInterval(lower, upper)
IntegerInterval(lower, upper)
```

以及 objective。

## 11.2 不把所有 bandwidth objective 强塞进 core

不同模型的 objective 不同：

- GWR strict LOOCV / AIC/AICc/BIC
- MGWR univariate partial-response search
- GTWR lambda + bandwidth
- MGTWR bandwidth + tau
- SGWR bandwidth + alpha
- SGTWR spatial + temporal + alpha
- STWR bandwidth + alpha + theta + ticks

因此：

- generic search algorithm → `core.optimization`
- bandwidth/neighbourhood representation → `core.weights`
- **objective → model/family private engine**

现有 public `BandwidthSelector` 类在兼容期可作为 standard-GWR selector facade，不再被视为“所有模型的统一 selector”。

## 11.3 MGWR reference-specific search 保留

MGWR adaptive univariate search若为了匹配 reference implementation需要特定 discrete golden-section 轨迹，可以拥有 model-specific search policy。

不能为了“统一 optimizer”破坏 reference equivalence。

---

# 12. Diagnostics 最终设计

## 12.1 `core.metrics` 拆分

最终：

### distribution-neutral metrics
`core.metrics`

- R²
- RMSE
- MAE
- 其他真正通用的误差指标

### Gaussian smoother diagnostics
`core.gaussian_diagnostics`

- trace(S)
- trace(S'S)
- ENP / EDF conventions
- Gaussian AIC/AICc/BIC
- sigma² conventions
- leverage/influence
- standardized residuals
- Cook's D
- local R² helper（若语义明确）

GWGLM Poisson/Binomial 的 log-likelihood/deviance 继续在 family-specific engine 中。

## 12.2 diagnostics 不再猜 attribute 名

现状存在：

```text
coords_train_
coords_
coords_summary_
eval_coords_
coords_data_
```

以及多组参数/权重别名。

最终 diagnostics 通过 private capability view/protocol 获取信息。

不创建巨型 universal result；建议使用小型、按能力分离的 typed views，例如：

```text
RegressionSurfaceView
ParameterInferenceView
TemporalView
WeightProviderView
StoredWeightComponentsView
```

具体 estimator 可以从自己内部状态构造这些 view。

## 12.3 plotting 绝不重新做模型数学

当前 plotting adapter 会重新构造 GWR distance matrix、调用 private `_weights_from_distances`、再独立做 SVD 条件数。

最终禁止：

- plotting 自己算 distance；
- plotting 自己算 kernel weights；
- plotting 自己做 local regression；
- plotting 自己定义 condition-number 数学。

plotting 只消费 diagnostics/view 的最终数组。

---

# 13. Result / export 最终设计

## 13.1 保留 model-specific public result

例如：

- `GWRPredictionResult`
- `GTWRPredictionResult`
- `GWGLMPredictionResult`
- `LGGWRPredictionResult`

它们具有不同的合法字段，不合并。

## 13.2 不把 generic `to_frame()` 塞进 regression base

当前 base 对 `intercept_ / coef_ / coords_train_ / local_r2_` 做假设。

最终 `to_frame()`：

- 由 public estimator 或其 model-specific result 实现；
- 可以调用共享的 column assembly helper；
- base 不假设参数形状。

## 13.3 统一的是命名协议，不是类

能够统一的公共列名尽量冻结：

```text
coord_0, coord_1
prediction / fitted
residual
intercept
coef_<feature>
se_<feature>
t_<feature> / z_<feature>
time（时空结果）
```

模型特有字段继续保留。

---

# 14. Execution / memory policy 最终设计

执行策略作为 private infrastructure 独立存在，不能进入 estimator inheritance。

概念上至少区分：

```text
BLOCK_STREAM
DENSE_CACHE
KNN_COMPRESSED
AUTO(memory budget)
```

这不是要求一次性实现四个公开选项，而是冻结设计边界：以后性能改造只能换 execution policy，不得重写 estimator class hierarchy。

## 14.1 GWR

冻结：

- 默认 bounded distance streaming；
- `compute_hat_matrix=False` 不创建 dense S；
- `compute_hat_matrix=True` 明确允许 dense S；
- bandwidth search streaming；
- 不恢复 n×n distance cache 作为默认。

## 14.2 MGWR

重复 backfitting 会反复读取相同空间距离。

因此不能机械复制 GWR streaming。

目标：

- small/medium data：dense/cache；
- large data：future memory-budget strategy；
- exact inference 自己有 chunking；
- `store_partial_hat_matrices=True` 属于明确 opt-in dense result。

## 14.3 GTWR

当前 lambda candidates 会重复构造 spatial/temporal/combined matrices，最终还保留三个 training matrices。

架构迁移阶段只保证数值不变；后续独立性能 PR再引入：

- base spatial/time component cache；
- combined distance按 lambda/tau 计算策略；
- optional storage；
- block path。

## 14.4 MGTWR

目标：

- spatial base distance可缓存；
- temporal base distance可缓存；
- coefficient/tau-specific combined distance尽量按需，不为每个候选长期保留；
- exact inference chunk policy独立。

## 14.5 ScalableGWR

冻结：

- cKDTree / neighbour-compressed / polynomial moment architecture是模型本身；
- **绝不能**为了 core 统一退化成 ordinary dense or streamed pairwise GWR。

## 14.6 SGWR / SGTWR / STWR

当前多个模型默认 `store_weights=True`，可能保留多个 dense weight matrices。

未来目标：

- provider 负责计算；
- storage 是 opt-in policy；
- 自动搜索不应因“需要评分”就永久保留每个候选的 dense matrix；
- 数值迁移和性能优化必须拆成不同 PR。

---

# 15. Public API 最终分级

API 分级现在就冻结，但真正删除/重命名可以到 0.2.0。

## Tier A — 稳定 end-user API

重点保持：

- `pygwrx.<Estimator>`
- `pygwrx.models.<Estimator>`
- public prediction result classes
- built-in datasets
- user-facing diagnostics/plotting/io
- documented built-in kernels

在架构迁移中，Tier A constructor、方法名、核心 fitted attrs、result columns原则上保持不变；有意变更必须写 migration note。

## Tier B — provisional advanced API

当前 `pygwrx.core` 中已有但过度暴露的：

- base classes
- low-level solver
- selector classes
- optimization classes
- distance helpers
- `DistanceCache`
- validation helpers
- hat matrix helper

0.1.x 不突然删除；文档标记 advanced/provisional。

0.2.0 统一做 deprecation/removal decision。

## Tier C — private implementation

本轮新架构的：

- `_protocols.py`
- `_execution.py`
- `_..._engine.py`
- internal result objects
- internal policy objects

**默认禁止进入 `__all__`。**

---

# 16. 数值验证分层

## Gate N0 — algebra unit tests

针对：

- distance
- kernel
- neighbourhood policy
- WLS
- optimizer
- Gaussian diagnostics

## Gate N1 — model characterization

每个 estimator 在迁移前必须冻结：

- constructor signature
- representative fit
- representative prediction/transform/classification/statistics
- required fitted attrs
- result frame columns
- failed-refit state clearing
- optional heavy-output flags

N1 是“当前行为合同”，不是独立外部真值。

## Gate N2 — independent/reference numerical validation

只要外部 reference 可获得，就在迁移前建立。

### 已达到高强度 N2

标准 GWR：已有 mgwr / GWmodel / spgwr，多 kernel/bandwidth/diagnostic/reference，必须全部阻塞。

### 较强但仍需整理成迁移 gate

GTWR：已有独立手算 distance/WLS、GWR reduction、CV search 等测试；迁移前将关键部分明确成 frozen numerical gate，并尽可能增加公共 Python GTWR / GWmodel fixture。

### 迁移前必须加强

- MGWR
- MGTWR
- GWGLM
- RGWR
- LCRGWR
- STWR
- GWPCA
- GWDA
- GWSS

按可获得 reference 优先采用：

- mgwr/spglm
- GWmodel
- published implementation
- published dataset/table

若外部 reference确实不可获得，至少建立：

- frozen characterization fixture；
- limiting-case identities；
- independent formula implementation；
- cross-model reduction identity。

## Gate N3 — invariant tests

跨模型必须保持的数学约束，例如：

- GTWR lambda=1 → matching GWR specification；
- SGWR alpha=1 → geographic-only limit；
- LGGWR separable attribute bandwidth=∞ → geographic-only limit；
- manual/auto storage flags不改变数值；
- no-hat storage仍保持 trace diagnostics；
- fixed/adaptive boundary语义明确。

---

# 17. 架构 characterization freeze：代码重构前必须先做

这是最终路线相对旧 master plan 最大的流程变化。

在任何大规模 base/core 改造前，先完成以下冻结 PR。

## A1 — Public API & capability snapshot

新增 machine-readable inventory，例如：

```text
architecture_contracts/estimators.json
```

每个 estimator 记录：

- public module
- constructor signature
- fit signature
- predict/transform/proba/statistics capability
- public result class
- key fitted attrs
- `to_frame()` / result columns
- optional dense outputs
- current public deprecations

CI 加测试，意外变更直接失败。

**MRO / concrete parent 不作为需要保持的 public contract**，因为本次重构明确要改变它。

## A2 — Fitted-state atomicity freeze

所有 estimator 建立失败重拟合测试：

1. 成功 fit；
2. 使用非法参数再次 fit；
3. 失败后不得留下“上一轮 fitted 状态与本轮部分新状态混合”的对象。

标准 GWR 已有，应推广。

## A3 — Model migration risk matrix

对 19 个模型标记：

- external reference强度；
- concrete inheritance依赖；
- dense O(n²)状态；
- custom search；
- custom solver；
- downstream diagnostics/plotting依赖。

此表进入 repo，后续 PR 不靠聊天记忆判断风险。

## A4 — Performance / memory baseline

建立 `benchmarks/` 或 `tools/benchmarks/`，至少记录：

- GWR fit manual bandwidth；
- GWR auto bandwidth；
- MGWR representative backfit；
- GTWR representative fit；
- MGTWR representative fit（小规模）；
- SGWR weight-heavy fit；
- ScalableGWR large-n representative。

性能数据默认不做严格跨平台 CI 阈值；CI 使用结构型 memory tests 防止明显的 O(n²) regression。

---

# 18. 最终实施顺序

下列顺序冻结。后续对话不得因为“某个模型看起来容易”跳阶段。

## Phase A — Safety freeze（先做，不改架构）

### A1. API/capability snapshot
### A2. fitted-state atomicity snapshot
### A3. 19-model risk matrix
### A4. performance/memory baseline harness
### A5. 修复明确的文案/契约错误，但不改变算法

已发现例子：GTWR 在正权重样本不足时警告“ridge regularized”，但当前调用的是未显式 ridge 的 rank-aware WLS；需要改为真实的 minimum-norm/rank warning，不能误导用户。

---

## Phase B — Core private spine（仍尽量不改 estimator 行为）

### B1. `_protocols.py`

只放 private structural Protocol：

- fitted lifecycle
- regression surface
- parameter inference
- temporal view
- multiscale view
- weight provider
- stored weight components

**不加入 public exports。**

### B2. `validation.py`

从 `utils.py` 拆出纯 validation。

旧 import 保留 compatibility re-export。

### B3. `distance.py`

拆普通 metric / block stream / cache primitives。

加入 `DistanceMetricSpec`（private first）和 metric params 支持。

### B4. `time.py`

抽出 GTWR/SGTWR 等通用时间规范化；不动具体 ST distance formula。

### B5. `weights.py`

实现显式 Fixed/Adaptive 内部 spec + `NeighbourhoodPolicy`。

先写语义测试，再迁移任何模型。

### B6. `solver.py`

建立 single canonical rank-aware WLS result path；public tuple API兼容。

### B7. `optimization.py`

从 `adaptive=True` 语义转成明确 continuous/integer domain；保留 compatibility wrapper。

### B8. `gaussian_diagnostics.py`

把 Gaussian smoother diagnostics 从 generic metrics 分开；旧 import compatibility wrapper保留。

---

## Phase C — GWR engine extraction（全项目黄金样板）

### C1. 建立 `_gwr_engine.py`

提取：

- GWR spatial weight orchestration；
- GWR bandwidth objective；
- Gaussian local fitting orchestration；
- inference collection；
- rank policy hook。

GWR public class仍保持现有 constructor/result API。

### C2. 全部 GWR numerical freeze gates通过

必须包括：

- 50+ external reference tests；
- deep hat/influence/Cook's D；
- rank-deficient behavior；
- prediction；
- streaming；
- bandwidth provenance；
- failed-refit atomicity。

### C3. 禁止新增新的 protected “给子类用” GWR 方法

GWR 不再作为其他 estimator 的 implementation framework。

---

## Phase D — 移除 concrete-estimator inheritance

顺序按依赖，不按模型名。

### D1. RGWR off GWR

使用 GWR engine + robust wrapper。

### D2. LCRGWR off GWR

共享 spatial infrastructure，但专用 LCR solver/condition semantics。

### D3. GWGLM off GWR

迁移前先完成 Poisson/Binomial numerical audit；其 `_RIDGE`/IWLS 稳定化策略不得因为脱离 GWR被暗改。

### D4. MGTWR off MGWR

**必须等 MGWR 与 MGTWR 自己的 freeze gate建立后再做。**

### D5. 删除生产代码中“public estimator 当 utility”的依赖

例如 SGTWR 的 GTWR time-converter 用 `core.time` 替换。

验收：生产代码 AST 测试禁止 class definition 的 base 是另一个 `pygwrx.models` public estimator。

---

## Phase E — 直接 BaseSpatialRegressor 使用者迁移到 explicit capabilities

目标模型：

- GWR
- MGWR
- GTWR
- GWLasso
- MixedGWR
- 以及后续纳入角色 base 的裸类

要求每个模型显式拥有自己的：

- constructor config；
- fitted state；
- weight/search/solver engine dependencies。

不再依赖 base 的 GWR-specific protected method。

---

## Phase F — Thin base cutover

只有当生产依赖扫描证明 base 中 GWR-specific 方法已无消费者时，才执行。

### F1. `BaseSpatialEstimator` 去除 distance/random/model math state
### F2. `BaseSpatialRegressor` 去除 kernel/bandwidth/local-regression state
### F3. role bases取消固定 fit signatures
### F4. `BaseSpatiotemporalRegressor` / `BaseMultiscaleRegressor` 降级为兼容 marker/deprecation path
### F5. 19 estimator 全部进入正确 role base

验收必须有 AST/contract test：

- base module 不 import kernels/bandwidth/solver/model metrics；
- base 不定义 `_compute_local_parameters` 等算法方法；
- public estimator MRO 不含另一个 public estimator。

---

## Phase G — 高风险模型逐族迁移

每个模型独立 PR，且先有 freeze gate。

推荐顺序：

1. GTWR
2. MGWR
3. MGTWR（在 D4 concrete inheritance 移除后继续 engine clean-up）
4. GWLasso
5. MixedGWR
6. SGWR
7. SGTWR
8. STWR
9. LGGWR
10. GRGWR
11. ScalableGWR（主要接生命周期/协议，不改核心算法）
12. GWPCA
13. GWDA
14. GWSS
15. BootstrapGWR

GWGLM 已在 D3 处理 concrete inheritance，可在本阶段继续内部 engine 清理。

### 重要：架构迁移和性能优化分 PR

例如 GTWR migration PR 只改变代码组织并保证完全等价；后续 GTWR performance PR 才改变 distance retention。

---

## Phase H — Diagnostics / plotting adapter 清理

### H1. 所有已迁模型提供 typed diagnostic/capability view
### H2. diagnostics 删除 `first_available(... alias ...)` 猜测
### H3. diagnostics weight view 改用 WeightProvider / StoredWeightComponents
### H4. plotting 删除 distance/kernel/SVD 再计算
### H5. 删除 model-specific legacy adapter分支

验收：

```text
plotting/ 不 import core.distance / core.kernels / core.solver
plotting/ 不调用 np.linalg.svd 做模型诊断
```

---

## Phase I — Performance policy implementation

此时类结构已稳定，性能改造不会再推动架构重写。

### I1. MGWR memory-budget cache strategy
### I2. GTWR distance component cache/stream strategy
### I3. MGTWR combined-distance execution strategy
### I4. SGWR/SGTWR/STWR optional dense weight storage
### I5. diagnostics on-demand weight blocks
### I6. benchmark report

保持每个模型的 numerical gate。

---

## Phase J — 0.2.0 public API consolidation

### J1. core public API tier cleanup / deprecation
### J2. `metric_params` 公共扩展（若本轮确认发布）
### J3. old compatibility re-export deprecation/removal
### J4. base marker/legacy class final处理
### J5. generated API docs / examples更新
### J6. migration guide from 0.1.x
### J7. version → 0.2.0

---

# 19. 每个 PR 的固定验收模板

任何架构 PR 必须在描述中填写：

```markdown
## Scope
- [ ] This PR changes one architecture responsibility only.

## Numerical semantics
- [ ] No estimator formula changed.
- [ ] Reference/characterization gates for affected models pass.
- [ ] No tolerance/reference fixture was weakened.

## API
- [ ] Tier-A constructor/method/result contract unchanged, or intentional change documented.
- [ ] No new private architecture object exported publicly.

## State lifecycle
- [ ] Successful refit works.
- [ ] Failed refit clears partial/previous fitted state according to contract.

## Memory/execution
- [ ] No accidental new dense n×n allocation.
- [ ] Any deliberate dense allocation is documented/opt-in/model-required.

## Architecture
- [ ] No public estimator inherits another public estimator.
- [ ] No new model math is placed in base classes.
- [ ] Diagnostics/plotting do not reimplement model math.

## Quality
- [ ] Black
- [ ] isort
- [ ] Ruff
- [ ] mypy architecture spine
- [ ] coverage
- [ ] build wheel/sdist
- [ ] docs strict
- [ ] security/SBOM
- [ ] platform matrix
```

---

# 20. 最终执行台账

> 每次合并 PR 后必须更新。没有记录的工作视为未完成。

| ID | 工作 | 状态 | PR | Merge SHA | 数值 gate | 备注 |
|---|---|---|---|---|---|---|
| Z0 | 第一次全项目审计 master plan | DONE | #30 | `da506c6f37b6154aa16105042464bc8960f8f036` | full CI + GWR refs | 历史审计文件保留 |
| Z1 | 三线独立二次审计 + 最终架构裁决 | DONE | #31 | `a7374d65296adc77a8a390ea9e97c76e8116352f` | docs-only | 最终架构宪章已冻结 |
| A1 | Public API & capability snapshot | DONE | #34 | `888f9ceac8fd2c988afb4a004056d5c523302290` | characterization | 19-estimator Public API / Capability Freeze |
| A2 | 19-model fitted-state atomicity freeze | DONE | #35 | `b655688f7201aaa9677fe153f2cbc15e6e63afb6` | characterization | failed-refit atomicity contract 覆盖 19 estimator |
| A3 | 19-model migration risk matrix | DONE | #36 | `dc150ef09fed370a45e5a6e62f14846979fff643` | n/a | machine-readable matrix + docs + contract tests；无 runtime 数值变更 |
| A4 | performance/memory baseline harness | PENDING | — | — | benchmark | **下一项代码工作**；不做 noisy CI timing gate |
| A5 | pre-refactor contract/message hygiene | PENDING | — | — | model tests | 含 GTWR misleading ridge warning |
| B1 | private `_protocols.py` | PENDING | — | — | no behavior | 不 public export |
| B2 | validation split | PENDING | — | — | core + all tests | compatibility re-export |
| B3 | distance split + metric spec | PENDING | — | — | core + GWR refs | keep streaming |
| B4 | time-axis utility | PENDING | — | — | GTWR/SGTWR tests | no ST formula change |
| B5 | weights + neighbourhood policies | PENDING | — | — | per-family semantics tests | 不强制一个 adaptive 规则 |
| B6 | canonical WLS result | PENDING | — | — | solver + GWR refs | pure algebra |
| B7 | generic search domains | PENDING | — | — | selector/optimizer tests | objective model-owned |
| B8 | Gaussian diagnostics split | PENDING | — | — | GWR refs | wrappers keep compatibility |
| C1 | GWR private engine extraction | PENDING | — | — | full GWR freeze | golden architecture sample |
| C2 | GWR engine numerical lock | PENDING | — | — | full reference | no relaxed tolerances |
| D1 | RGWR remove concrete GWR inheritance | PENDING | — | — | RGWR freeze | composition |
| D2 | LCRGWR remove concrete GWR inheritance | PENDING | — | — | LCR freeze | specialized solver |
| D3 | GWGLM remove concrete GWR inheritance | PENDING | — | — | family refs | audit IWLS/ridge first |
| D4 | MGTWR remove concrete MGWR inheritance | PENDING | — | — | MGWR+MGTWR freeze | multiscale composition |
| D5 | remove public-estimator-as-utility uses | PENDING | — | — | relevant models | e.g. SGTWR time conversion |
| E1 | direct regressor consumers explicit capability migration | PENDING | — | — | model gates | before thin base |
| F1 | thin BaseSpatialEstimator | PENDING | — | — | all contracts | no distance/random math config |
| F2 | thin role bases / remove algorithm methods | PENDING | — | — | all contracts | no fixed fit signature |
| F3 | role-base adoption all 19 estimators | PENDING | — | — | all contracts | no concrete MRO |
| G1+ | remaining model-by-model engine migrations | PENDING | — | — | per model | one model/family per PR |
| H1+ | diagnostics/plotting protocol migration | PENDING | — | — | diagnostics/plotting | remove guessing/recompute |
| I1+ | performance execution policies | PENDING | — | — | numerical + benchmark | after class architecture stable |
| J1+ | 0.2.0 API consolidation | PENDING | — | — | release matrix | final cleanup |

---

# 21. 三条审计线的最终裁决摘要

## 21.1 数值/统计线

**结论：允许统一基础设施，不允许统一估计量。**

可以统一：

- numeric validation
- ordinary distance metrics
- time-axis parsing
- kernel functions
- explicit neighbourhood mechanisms
- WLS linear algebra
- generic optimization algorithms
- Gaussian smoother formulas

不可被同一个实现强行吞并：

- LCR penalty semantics
- Lasso
- GLM IWLS
- MGWR/MGTWR backfitting
- STWR temporal effect
- Mixed partial regression
- ScaGWR compressed estimator
- latent geometry optimization
- regime discovery
- local PCA / DA / summary statistics

## 21.2 软件架构线

**结论：当前最大风险不是重复几行代码，而是 concrete inheritance + implicit fitted-state contracts。**

最终措施：

- root 极薄；
- public estimator 无互相继承；
- role base 无算法；
- capability 用 private Protocol/component；
- result model-specific；
- 下游通过 typed view，而不是 `Any/getattr`。

## 21.3 性能/内存线

**结论：执行策略必须与数学架构分离。**

- GWR streaming是正确默认；
- MGWR重复 backfit需要 cache思维；
- GTWR/MGTWR有 ST component reuse；
- SGWR/SGTWR dense weights需可选；
- ScalableGWR kNN/compressed path必须独立保留。

因此性能优化不再需要改变 estimator hierarchy。

---

# 22. 已发现、但必须与架构 PR 分开的具体债务

以下记录进入 backlog，禁止“顺便”塞进不相关架构 PR：

1. GTWR warning 声称 local solution “ridge regularized”，实际当前 shared WLS 默认 ridge=0；修正文案并增加 rank diagnostics。
2. GWGLM 内部存在显式 `_RIDGE = 1e-8` 数值策略；脱离 GWR inheritance 前必须独立确认其统计/数值意图。
3. MGWR 有 vectorized adaptive weight path 与 reference-specific adaptive search；不能被普通 GWR row helper简单替换。
4. GWPCA/GWDA/GWSS adaptive boundary/tie 规则需要先建立 reference gate，再决定共享哪部分 helper。
5. SGWR alpha search materialize full combined weights/hat matrices，且默认保存多份 dense weights；未来单独 performance PR。
6. plotting `_compute_gwr_condition_numbers` 重新 materialize distance matrix和独立 SVD；在 H 阶段删除。
7. diagnostics alias guessing说明 fitted-state/view protocol尚未统一；在 H 阶段删除。
8. current `pygwrx.core.__all__` 过度公开 low-level internals；0.1.x先分类，0.2.0处理。
9. current root/base 的 `distance_metric` 所有权错误；F 阶段移除。
10. current generic optimizer 的 `adaptive` 参数把统计语义泄漏进搜索算法；B7处理。

---

# 23. 什么情况才允许修改本最终设计

本文件不是“永远不能改代码”，而是阻止无证据的架构漂移。

只有以下情况可以提出新的架构 ADR：

1. 独立 reference 表明某个冻结接口会导致错误统计结果；
2. 一个新模型具有当前 capability体系无法表达的、可证明的新数学对象；
3. profiling证明当前 execution abstraction阻止关键性能优化，而且无法在 policy层解决；
4. Python/NumPy/SciPy 的重大上游 API 变化迫使结构改变；
5. public API 发布后的真实兼容需求。

不能作为理由：

- “这个 PR 写起来更方便”；
- “可以少几行代码”；
- “某个类名更好看”；
- “先继承 GWR 以后再拆”；
- “先把所有模型统一了再测试”；
- “性能可能更快”但无 benchmark；
- “大概与 reference 一样”。

---

# 24. 本次最终冻结后的第一步

**不要立即开始重写 base。**

下一项工作固定为：

> **A1 — 建立 19 个 estimator 的 Public API & Capability Snapshot。**

原因：

- 它把用户真正依赖的 contract 与本次有意改变的内部 MRO 分开；
- 后续每个 PR 都能知道自己是否意外破坏 API；
- 它是避免“改到一半才发现另一个模型依赖某属性”的最低成本保险。

A1 完成前，不进入 B1，不迁移任何模型。

---

# 25. 最终一句话架构

> **pyGWRx 0.2 的目标不是一棵“大一统 GWR 继承树”，而是：极薄 estimator shell + 私有 capability protocols + 模型无关数值原语 + 模型专用 engines + 独立 execution policies + model-specific public results。**

这样以后增加新模型时，选择/组合所需能力即可；优化距离与内存时，只改 execution/provider；修改诊断显示时，只改 view/downstream；不会再因为一个模型的特殊公式把整棵继承树推倒重来。
