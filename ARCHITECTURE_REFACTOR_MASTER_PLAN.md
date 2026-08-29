# pyGWRx 全项目架构审计与重构主计划

> **用途**：这是 pyGWRx 下一阶段大型架构重构的唯一主计划（master plan）与跨对话交接文档。
>
> **审计基线**：`main` @ `241246b9912466121956937ace068047d354576f`
>
> **审计日期**：2026-08-29
>
> **当前软件版本**：`0.1.2`（Alpha）
>
> **重要说明**：本文件只冻结“架构重构方向、执行顺序和验收规则”，不冻结未来每一个内部类名。任何实际改造开始前，都必须重新读取当时的 `main`，不能仅凭本文件中的旧代码片段直接修改。

---

## 0. 给后续对话 / 后续开发者的第一条指令

如果你是在新的 ChatGPT 对话、Codex 会话或新的开发环境中继续本项目，请按下面顺序开始：

1. 读取本文件：`ARCHITECTURE_REFACTOR_MASTER_PLAN.md`。
2. 读取标准 GWR 数值验证总报告：`validation_results/gwr/GWR_VALIDATION_EVIDENCE.md`。
3. 获取 GitHub `main` 当前最新 SHA，**不要假设仍然是本文件记录的基线 SHA**。
4. 查看本文件第 17 节“执行台账”，确定上一阶段已经完成到哪一步。
5. 只执行“下一项 pending 工作”，不要跨多个阶段做 mega-PR。
6. 每完成一个 PR 后：
   - 更新本文件执行台账；
   - 写入 PR 号、merge SHA、验证结果；
   - 再开始下一阶段。
7. 所有涉及标准 `GWR` 的内部重构，都必须保持已有 50 个 external-reference tests 通过；**不得为了让架构重构通过而放宽 reference tolerance、删除 reference assertion 或修改 frozen reference 数据。**

本文件在架构问题上**取代**旧的 `HANDOFF_NEXT_CONVERSATION.md`。旧 handoff 中关于文档样式、模型说明等仍可作为历史背景，但如果其架构决策与本文件冲突，以本文件和最新 `main` 为准。

---

# 1. 审计结论摘要

当前 pyGWRx 已经具备非常完整的软件外观：19 个 estimator、统一的 `core/`、`diagnostics/`、`plotting/`、`io/`、完整 MkDocs、跨平台 CI、typed API gate、wheel/sdist、安全扫描以及标准 GWR 的独立软件数值验证。

但是从内部架构看，当前项目处于一个典型的“**半统一架构**”状态：

- `core/` 看起来是统一数值后端，但只有一部分模型真正按同一套逻辑使用；
- `BaseSpatialRegressor` 名字很通用，但内部包含了大量标准 GWR 专属的 kernel / bandwidth / local regression 行为；
- 一些模型通过继承 `GWR` 复用代码，即使它们在统计模型意义上并不是严格的 GWR 子类；
- 另一些模型完全不进入 base hierarchy；
- `diagnostics/` 为了兼容不同模型，大量依赖 `Any + getattr + 属性别名猜测`；
- `plotting/` 同样需要 adapter，并且有时重新计算距离、权重或条件数；
- adaptive bandwidth 转换、weight 生成、distance policy、search bounds 等概念在多个层级重复出现；
- `utils.py` 同时承担距离、验证、GeoPandas、内存建议、chunk 等多种无关职责；
- `core.metrics` 实际是 Gaussian GWR 诊断，却以通用 `metrics` 命名；
- 目前 mypy 只覆盖少量文件，尚未形成真正的 typed architectural spine。

因此，本项目下一阶段的目标不应是“继续给每个模型加功能”，而应是：

> **先把 pyGWRx 重构为一个真正由统一核心能力驱动、模型数学定义与软件基础设施明确分层的空间加权建模框架。**

核心原则只有一句：

> **Inheritance for lifecycle/contracts; composition for numerical capabilities.**  
> **继承用于生命周期与接口契约，组合用于数值能力与模型数学。**

---

# 2. 本次审计范围

本轮不是标准 GWR 数值审计（标准 GWR 已完成独立验证），而是软件架构审计。已检查的主要范围：

```text
src/pygwrx/
├── __init__.py
├── _optional.py
├── core/
│   ├── __init__.py
│   ├── base.py
│   ├── bandwidth.py
│   ├── kernels.py
│   ├── metrics.py
│   ├── optimization.py
│   ├── solver.py
│   ├── utils.py
│   ├── _summary.py
│   └── _legacy_solver.py
├── models/                     # 19 个 estimator
├── diagnostics/
├── plotting/
├── io/
└── data/

tests/
tools/
docs/
examples/
.github/workflows/
pyproject.toml
PROJECT_STRUCTURE.md
HANDOFF_NEXT_CONVERSATION.md
```

同时核查了：

- 当前 19 个 estimator 的真实继承关系；
- `core` 能力实际被哪些模型复用；
- 标准 GWR 的新 streaming / rank-aware 数值路径与旧公共 helper 是否冲突；
- diagnostics / plotting 如何适配不同模型；
- public API inventory；
- mypy 覆盖范围；
- CI 与 reference test 结构；
- data / io / optional dependency 分层；
- 标准 GWR frozen evidence 与未来重构的兼容要求。

本轮**没有**重新证明每个非 GWR 模型的学术公式正确性。因此，后续每个模型迁移时仍需执行“源码 + 原论文 / 官方实现 + reference test”的模型级验证。

---

# 3. 当前项目真实架构

## 3.1 公开层面

当前 `PROJECT_STRUCTURE.md` 描述的 release-facing 结构已经比较成熟：

- `models`：19 个 estimator，另有 7 个 prediction result class；
- `core`：51 个 public symbols；
- `diagnostics`：23 个 public symbols；
- `plotting`：56 个 public symbols；
- `io`：17 个 public symbols；
- 总 public surface：173 个 symbols。

问题不在“功能不够多”，而在于**公开出来的 core 能力太多，同时内部又没有真正统一使用这些能力**。

尤其是以下内容现在都是 public core：

- base classes / mixins；
- BandwidthSelector 及其具体类；
- GoldenSectionSearch / BrentSearch；
- `local_regression`；
- `compute_hat_matrix`；
- `DistanceCache`；
- 各类低层 helper。

这会让未来重构更困难，因此 0.2.0 前必须重新明确：

1. **稳定用户 API**；
2. **高级/开发者 API**；
3. **纯内部 private API**。

---

## 3.2 当前 base hierarchy

当前 `core/base.py` 的设计树是：

```text
BaseSpatialEstimator
│
├── BaseSpatialRegressor
│   ├── BaseSpatiotemporalRegressor
│   └── BaseMultiscaleRegressor
│
├── BaseSpatialClassifier
├── BaseSpatialTransformer
├── BaseSpatialStatistics
└── BaseSpatialInference
```

设计思想本身没有问题，但实际模型没有完全进入这棵树。

### 当前真实模型继承关系

```text
BaseSpatialEstimator
│
└── BaseSpatialRegressor
    │
    ├── GWR
    │   ├── RGWR
    │   ├── LCRGWR
    │   └── GWGLM
    │
    ├── GWLasso
    ├── MixedGWR
    │
    ├── BaseSpatiotemporalRegressor
    │   └── GTWR
    │
    └── BaseMultiscaleRegressor
        └── MGWR
            └── MGTWR

未进入统一 base tree：

STWR
GWPCA
GWDA
GWSS
ScalableGWR
BootstrapGWR
SGWR
SGTWR
LGGWR
GRGWR
```

### 结论

当前项目不是“统一继承体系”，而是：

> 一部分模型统一继承 + 一部分模型独立 class + 一部分模型为了代码复用继承概念上并不完全合适的父类。

---

# 4. Base 层核心问题

## 4.1 `BaseSpatialRegressor` 过厚

当前 `BaseSpatialRegressor` 不只做 regression estimator 的生命周期，还直接知道：

```text
kernel
bandwidth
bandwidth_method
adaptive
bandwidth_range
optimization_method
fit_intercept
```

还保存：

```text
X_train_
y_train_
coords_train_
times_train_
context_train_
coef_
intercept_
fitted_values_
residuals_
diagnostics_
local_r2_
bandwidth_
kernel_func_
hat_matrix_
```

并实现：

```text
score()
get_diagnostics()
to_frame()
to_geodataframe()
_predict_basic()
_compute_local_parameters()
```

特别是 `_compute_local_parameters()` 会直接调用：

```text
kernel -> local_regression -> local WLS
```

这已经是**标准 GWR 算法行为**，不应该存在于一个通用名称的 `BaseSpatialRegressor` 中。

## 4.2 Base 与标准 GWR 已出现两套数值语义

这是当前最明确的架构缺陷之一。

现在标准 `GWR` 已经过本轮重构，预测使用：

- bounded distance streaming；
- `_weighted_least_squares_details()`；
- local rank；
- condition number；
- rank-deficient minimum-norm coefficient；
- rank-deficient inference = NaN。

而 `BaseSpatialRegressor._compute_local_parameters()` 仍走公共 `core.solver.local_regression()`，后者会一次构造 target×train distance matrix，且没有完全相同的 rich local result contract。

因此目前存在：

```text
BaseSpatialRegressor generic local prediction
                VS
GWR authoritative local prediction
```

这种结构不能继续扩散到其他模型。

## 4.3 不应重新引入巨大的 `BaseGWR`

历史上已经移除了旧 GWR-specific base。下一阶段也**不应该**重新创建一个把所有算法都塞进去的 `BaseGWR`。

正确方向是：

- base 变薄；
- 数值能力放入 pure core；
- 模型通过组合使用 core capability。

---

# 5. Core 层逐文件审计

## 5.1 `core/kernels.py` — 当前最健康的模块

这是最接近目标架构的文件：

```text
distance array + bandwidth
        ↓
pure kernel
        ↓
weight array
```

内置：

- Gaussian
- Bisquare
- Exponential
- Tricube
- Boxcar
- kernel registry / `get_kernel_function`

### 决策

**保留并作为其他 core 模块的设计范例。**

但要保持边界：

- SGWR 的 geographic + similarity 混合不是 basic kernel；
- SGTWR 的 spatiotemporal + similarity 不是 basic kernel；
- robust residual weight 也不是 basic spatial kernel。

这些组合规则属于模型或 weight-composition 层。

---

## 5.2 `core/utils.py` — 必须拆分

当前 `utils.py` 同时承担：

- coordinate validation；
- Euclidean/Manhattan/Chebyshev/Minkowski/Haversine；
- `compute_distance_matrix`；
- bounded block/row iterator；
- `DistanceCache`；
- intercept；
- chunk；
- GeoPandas coordinate extraction；
- data validation。

这是典型的“utility junk drawer”。

### 明确问题 1：distance config 无法从 estimator 完整表达

底层 `compute_distance_matrix` 支持：

```text
minkowski(p=...)
haversine(radius=...)
```

但 estimator 目前通常只有：

```python
distance_metric="euclidean"
```

没有统一的 `distance_kwargs` / `DistanceSpec`。

因此例如：

- `distance_metric="minkowski"` 实际只能走默认 `p=2`；
- Haversine radius 也不能从模型 API 正式配置。

这属于 API / core 设计缺口。

### 明确问题 2：`DistanceCache` 已与标准 GWR 新默认路径脱节

`DistanceCache` 会推荐：

> Precompute and cache the distance matrix.

但标准 GWR 和标准 GWR bandwidth selector 已在 #25–#27 后改为 bounded streaming。

因此当前 public `DistanceCache` 更像历史 advisor，而不是实际算法策略。

### 目标拆分

建议最终拆为：

```text
core/validation.py
core/distance.py
core/arrays.py         # 如确有必要，仅放极少量 array helper
```

其中 `core/distance.py` 应提供：

```text
DistanceSpec
pairwise_distance(...)
iter_distance_blocks(...)
iter_distance_rows(...)
nearest-neighbour / kNN strategy interface
```

并明确支持不同 execution strategy：

```text
dense
block-streamed
kNN / KDTree
custom / learned
spatiotemporal (由模型或 specialized strategy 扩展)
```

**注意：统一的是 distance contract，不是强制所有模型使用同一种实现。**

例如 ScalableGWR 使用 KDTree 是合理的；LGGWR 使用 learned geometry 也是合理的。

---

## 5.3 `core/solver.py` — 数值核心好，但职责混杂

### 已经非常好的部分

当前 `_weighted_least_squares_details()` 是标准 GWR 经过验证的 rank-aware 单次 SVD WLS 后端：

- unpenalized default；
- minimum-norm solution；
- inverse normal operator；
- numerical rank；
- singular values；
- condition number。

这是未来 Gaussian local linear model 应优先复用的 canonical primitive。

### 当前问题

`solver.py` 同时还做：

- X/y/weight validation；
- coordinate validation；
- kernel validation；
- adaptive bandwidth conversion；
- distance matrix；
- local-regression orchestration；
- hat matrix。

尤其公共 `local_regression()` 目前仍会构造完整 target×train distance matrix，而标准 GWR 自己已经走 streaming。

因此：

> solver 既是“线性代数求解器”，又是“空间 GWR orchestrator”。

### 目标

solver 只应该解决数值问题：

```text
(X, y, weights)
       ↓
LocalSolveResult
```

推荐 canonical result：

```python
LocalSolveResult(
    beta,
    inverse_normal,
    rank,
    singular_values,
    condition_number,
)
```

距离、kernel、bandwidth 不应由 solver 决定。

### 明确禁止

不能为了统一而强迫：

- GWGLM 的 IWLS；
- GWLasso 的 L1 solver；
- LCRGWR 的 local ridge；

改成普通 WLS。

这些是不同 solver strategy。

---

## 5.4 `core/bandwidth.py` — 标准 GWR 很强，但并不是真正 generic

目前该模块名称是“Bandwidth selection for geographically weighted models”，但内部实际硬编码了标准 Gaussian GWR 的很多语义：

- local WLS；
- leave-one-out；
- hat row；
- GWR AIC/AICc/BIC；
- GWR 可估计性判断。

因此它并不是可直接用于所有空间模型的真正通用 selector。

### 重复问题

当前 adaptive bandwidth conversion 至少存在于：

- `core/bandwidth.py`；
- `core/solver.py`；
- `diagnostics/collinearity.py`；
- 某些 model-specific code。

compact kernel 对第 k 个邻居边界的 `nextafter` 处理也有重复。

### 性能问题

`_fit_local_model()` 当前会先：

```python
np.linalg.matrix_rank(Xw)
```

然后再调用 WLS，而 WLS 内部又 SVD。

即一个 candidate/local fit 发生重复 decomposition。

### 目标拆分

未来应该拆成两层：

```text
Generic search/domain machinery
           +
Model-specific bandwidth objective
```

例如：

```text
SearchEngine
  ├─ exhaustive integer
  ├─ grid
  ├─ golden section
  └─ Brent

GWRBandwidthObjective
  ├─ CV
  ├─ AIC
  ├─ AICc
  └─ BIC
```

GTWR、GWDA、SGWR 等可以复用 search engine，但不需要假装它们使用同一个 GWR objective。

---

## 5.5 `core/optimization.py` — 通用优化与 bandwidth policy 混在一起

当前模块的 GoldenSection / Brent 本身属于 generic optimization。

但同时存在 bandwidth-specific `auto_bounds()` 等策略；而 `core/bandwidth.py` 又拥有另一套 automatic range。

这意味着：

> automatic bounds 有两个潜在 owner。

### 决策

`core/optimization.py` 最终只负责：

```text
minimize(objective, domain)
```

不应知道：

```text
GWR
bandwidth
coordinate distance
minimum neighbours
```

bandwidth domain 由 bandwidth/domain layer 负责。

---

## 5.6 `core/metrics.py` — 命名过于泛化

模块 docstring 已经明确：

> Diagnostic metrics for Gaussian geographically weighted models.

但是文件名叫 `metrics.py`，对外又公开：

- AIC
- AICc
- BIC
- local R²
- trace statistics
- ENP/EDF
- `compute_diagnostics`

这会诱导其他模型复用并不适合自己的 Gaussian 诊断。

### 目标

建议至少逻辑上拆为：

```text
core/metrics.py
    RSS / RMSE / MAE / R² 等真正通用统计

core/gaussian_diagnostics.py
    trace(S)
    trace(S'S)
    ENP / EDF
    Gaussian AIC / AICc / BIC
    GWR local R²
```

而：

- GLM family diagnostics；
- classification diagnostics；
- PCA diagnostics；

保持 model/family-specific。

---

## 5.7 `_summary.py`

当前一些模型调用统一 formatter，一些模型（例如标准 GWR）自己构造较完整 summary。

最终原则应该是：

```text
模型决定“输出哪些统计含义”
formatter 决定“怎么显示”
```

不能让 formatter 决定统计内容，也不应让每个 model 重新拼版式。

---

# 6. Diagnostics 层审计：当前是“适配器统一”，不是“协议统一”

这是本轮发现的第二个最重要架构问题。

当前 diagnostics 为了支持不同模型，不得不大量使用：

```python
Any
getattr(...)
first_available(...)
多个属性别名
```

例如为了找 training coordinates，需要依次尝试：

```text
coords_train_
coords_
coords_summary_
eval_coords_
coords_data_
coords_stages_[-1]
```

为了找统计量又需要兼容：

```text
r2 / R2
adj_r2 / adjusted_r2 / adj_R2
trace_S / tr_S / effective_params
...
```

推断层还需要尝试：

```text
coef_t_
coef_z_
parameter_t_values_
parameter_z_values_
t_values_
parameter_standard_errors_
standard_errors_
```

权重层需要尝试：

```text
spatial_weights_
temporal_weights_
spatiotemporal_weights_
similarity_weights_
combined_weights_
weights_
```

### 这说明什么？

不是 diagnostics 写得差，而是：

> **上游模型没有稳定的 fitted-result protocol。**

于是 diagnostics 只能“猜模型是什么结构”。

### 更严重的问题

`LocalCollinearityDiagnostics` 会自己：

- 重建 full distance matrix；
- 重建 adaptive bandwidth；
- 重建 kernel weights；
- 再算局部 SVD。

`plotting/_adapters.py` 在缺少 condition number 时也会重新构造 distance + weights + SVD。

这导致下游层可能与实际 fitted model 数值语义产生漂移。

### 目标

迁移后的 model 应提供明确、typed 的 view/protocol，例如概念上：

```text
FittedSpatialProtocol
LocalParameterProtocol
DiagnosticProtocol
WeightProviderProtocol
TemporalProtocol
```

或明确方法：

```text
get_training_view()
get_parameter_view()
get_diagnostic_view()
get_weight_view()
```

具体命名在实施 PR 中决定，但原则不变：

> 新代码不得继续新增 model-specific getattr alias guessing。

旧模型未迁移前可以保留 legacy fallback，迁移完后删除。

---

# 7. Plotting 层审计

当前 plotting 已有 `_adapters.py`，这是正确方向，但 adapter 目前依赖 model internals，并在部分情况下自己执行计算。

### 当前问题

plotting 不应该负责：

- distance matrix；
- spatial weight reconstruction；
- SVD condition number；
- statistical inference；
- model-specific fallback calculation。

plotting 应该只做：

```text
normalized view/dataframe
       ↓
visual rendering
```

### 最终目标

```text
model/core
   ↓
diagnostic/result view
   ↓
plotting
```

而不是：

```text
plotting
   ↓
getattr model internals
   ↓
重新计算统计量
```

---

# 8. IO / Data / Dependency 层审计

## 8.1 `io/` 总体结构较清楚

`io/data.py`：

- CSV / Shapefile / GeoJSON / GPKG / Parquet；
- array extraction；
- GeoDataFrame；
- save/export。

`io/datasets.py`：

- bundled dataset registry；
- provenance；
- aliases；
- canonical features/response/coords。

这部分不是优先大问题。

## 8.2 GeoPandas 相关逻辑有跨层重复

`core.utils` 也包含 GeoPandas coordinate extraction，而 `io.data` 已经负责 geospatial IO。

最终应把文件/GeoPandas/geometry 相关逻辑集中在 `io/`。

core 只接受已经规范化的 numeric coordinates。

## 8.3 dependency policy 后续需要单独审计

当前 mandatory dependencies 包括：

```text
numpy
scipy
pandas
matplotlib
geopandas
shapely
```

同时 `_optional.py` 又存在 optional-dependency machinery，`scikit-learn` 与 `pyarrow` 是 extras。

未来可以讨论：

```text
minimal numerical core
+ [plot]
+ [geo]
+ [ml]
+ [parquet]
```

但**不要在第一轮架构重构中同时改 dependency policy**。这属于后期 Phase H。

---

# 9. Typing / CI / Tests 审计

## 9.1 CI 是本次大型重构的重要优势

当前 CI 已有：

- Black；
- isort；
- Ruff；
- mypy gate；
- generated docs diff gate；
- Ubuntu / Windows / macOS；
- Python 3.11–3.14；
- coverage；
- minimum dependencies；
- wheel/sdist isolated installation；
- independent reference tests；
- security/SBOM。

这是非常好的重构安全网。

## 9.2 当前 mypy 不是全项目 typed guarantee

`pyproject.toml` 只对少量文件进行 mypy：

- 部分 core；
- diagnostics `_utils`；
- io；
- MGTWR；
- plotting validation 等。

并且 `follow_imports="skip"`。

因此“typed public surface”目前不能理解为整个内部架构都有强类型约束。

### 目标

每个新 architecture contract 文件必须立即加入 mypy。

迁移顺序：

```text
core contracts/types
→ canonical GWR
→ diagnostics views
→ 每迁一个 model，就把那个 model 纳入严格类型检查
```

最终再评估是否收紧 `follow_imports`。

## 9.3 现有 base hierarchy test 太弱

当前测试只检查 GWR、GWLasso、MixedGWR、MGWR 是否是 `BaseSpatialRegressor`。

未来需要升级成 architecture contract tests，而不是只看 `issubclass`。

---

# 10. 标准 GWR：整个重构的数值基准

标准 `GWR` 是下一阶段所有架构改造的“golden implementation”。

截至本文件基线：

- 50 个 external-reference tests；
- mgwr 2.2.1；
- GWmodel；
- spgwr；
- synthetic + Columbus；
- fixed/adaptive；
- Gaussian/bisquare；
- bandwidth full criterion trace；
- coefficients；
- fitted/residuals；
- local R²；
- SE/t；
- hat matrix；
- influence；
- standardized residuals；
- Cook’s D；
- held-out prediction。

因此，任何架构 PR 如果涉及 GWR：

### 禁止

- 修改 frozen reference fixture；
- 放宽 reference tolerance；
- 删除 reference test；
- 用“架构更漂亮”为理由接受 GWR 数值变化。

### 必须

- 50 reference tests 全绿；
- freeze-contract tests 全绿；
- streaming tests 全绿；
- rank-inference tests 全绿；
- bandwidth provenance/trace 全绿。

如果新架构不能保持这些结果，则优先修架构，不是修 reference。

---

# 11. 目标架构

## 11.1 第一原则

```text
Base classes = 生命周期 / 数据契约 / common fitted state
Core         = pure numerical capabilities
Models       = 统计模型数学定义 + orchestration
Diagnostics  = 读取标准化 result/capability，不猜 internals
Plotting     = 只渲染，不重新算模型
IO           = 文件 / GeoDataFrame / dataset 边界
```

## 11.2 目标顶层

```text
BaseSpatialEstimator
│
├── BaseSpatialRegressor
├── BaseSpatialClassifier
├── BaseSpatialTransformer
├── BaseSpatialStatistics
└── BaseSpatialInference
```

这些 base 必须是“薄 base”。

### Base 可以做

- fitted lifecycle；
- atomic failed-fit reset；
- `n_samples_` / `n_features_in_`；
- feature names；
- common input normalization；
- stable training/result view hook；
- common export contract。

### Base 不应该做

- kernel selection；
- GWR bandwidth；
- local WLS；
- spatial weight formula；
- hat matrix；
- GWR prediction algorithm。

---

## 11.3 目标 core（概念布局）

最终结构可以类似：

```text
core/
├── base.py
├── contracts.py
├── validation.py
├── distance.py
├── kernels.py
├── weights.py
├── optimization.py
├── bandwidth.py
├── solver.py
├── metrics.py
├── gaussian_diagnostics.py
├── results.py
└── _summary.py
```

不要求一次全部重命名；这是目标职责图，而不是要求某个 PR 一次完成文件移动。

---

## 11.4 建议的基础 contract

### `DistanceSpec`

目标解决当前只有 string metric 的问题：

```python
DistanceSpec(
    metric="minkowski",
    kwargs={"p": 1.5},
)
```

或：

```python
DistanceSpec(
    metric="haversine",
    kwargs={"radius": 6371.0},
)
```

并允许 specialized/custom strategy。

### `LocalSolveResult`

统一 Gaussian WLS rich result：

```python
LocalSolveResult(
    beta=...,
    inverse_normal=...,
    rank=...,
    singular_values=...,
    condition_number=...,
)
```

### `SearchResult`

统一：

```python
SearchResult(
    value=...,
    score=...,
    trace=...,
    search_range=...,
    boundary_solution=...,
    converged=...,
    evaluations=...,
    message=...,
)
```

### Fitted/result protocol

必须至少能稳定表达：

```text
is_fitted
n_samples
feature names
coordinates（若存在）
training response（若存在）
fitted values（若存在）
parameter surfaces（若存在）
diagnostics
weight components（若模型选择存储）
temporal axis（若存在）
```

这里建议优先使用 `typing.Protocol` 做 downstream capability contract，而不是继续增加复杂多继承。

---

# 12. 19 个模型的架构迁移矩阵

符号：

- ✅：结构相对合理，可保留方向；
- ⚠️：部分复用但需要重构；
- ❌：未进入统一协议；
- **高风险**：不能和算法改动同时进行。

| Model | 当前父类 | 当前主要问题 | 目标方向 | 风险 |
|---|---|---|---|---|
| **GWR** | `BaseSpatialRegressor` | base 与 GWR 有两套 local prediction；distance config 不完整 | 新架构第一 canonical adopter；GWR 显式拥有 kernel/bandwidth/orchestration | 中，但有 50 refs 保护 |
| **RGWR** | `GWR` | 依赖 GWR protected behavior | 可继续作为真 GWR variant；逐步改为显式 GWR calibration capability | 中 |
| **LCRGWR** | `GWR` | 自己重建 local fit/distance；ridge solver 特有 | 共享 GWR orchestration + 专门 LCR solver strategy | 中高 |
| **GWGLM** | `GWR` | 继承主要为了复用；统计意义并非标准 GWR 子类 | 最终直接基于 thin `BaseSpatialRegressor` + local IWLS engine | 高 |
| **GWLasso** | `BaseSpatialRegressor` | base 过厚；search/distance 可共享 | thin regressor + shared distance/kernel/search + L1 solver | 中 |
| **MixedGWR** | `BaseSpatialRegressor` | private mixed core 与公共 core 分层不清 | 保留 partial-regression algorithm，接入 shared contracts/distance/search | 中 |
| **MGWR** | `BaseMultiscaleRegressor` | dense distance / inference memory；core solver contract不统一 | shared distance strategy + multiscale backfitting engine | **高风险** |
| **GTWR** | `BaseSpatiotemporalRegressor` | 自己的 search / dense space-time matrices | shared generic search + specialized ST distance engine | **高风险** |
| **MGTWR** | `MGWR` | 概念上=multiscale+spatiotemporal，但只体现 MGWR 继承 | 最终组合 multiscale engine + spatiotemporal distance；弱化对 MGWR 具体类继承 | **最高风险** |
| **STWR** | bare | staged API 与 base contract完全分离 | thin estimator/regressor lifecycle + staged capability protocol | 高 |
| **SGWR** | bare | geography/similarity 自己组织；diagnostics靠 adapter | thin regressor + shared spatial primitives + model-specific similarity composer | 中高 |
| **SGTWR** | bare | 复用 GTWR 但非统一 base；组合权重特有 | spatiotemporal regressor capability + similarity composer | 高 |
| **ScalableGWR** | bare | 自成 KDTree/compressed engine | 接入 thin lifecycle/result protocol；**保留 specialized scalable backend** | 中 |
| **LGGWR** | bare | learned distance 自成体系 | thin regressor + custom/learned DistanceStrategy | 中高 |
| **GRGWR** | bare | regime discovery 自成体系 | thin regressor/result protocol；GWR 作为 composition | 中高 |
| **GWPCA** | bare | 已有 `BaseSpatialTransformer` 但不用 | 迁入 thin `BaseSpatialTransformer` | 中 |
| **GWDA** | bare | 已有 `BaseSpatialClassifier` 但不用 | 迁入 thin `BaseSpatialClassifier` | 中 |
| **GWSS** | bare | 已有 `BaseSpatialStatistics` 但不用 | 迁入 thin `BaseSpatialStatistics` | 低中 |
| **BootstrapGWR** | bare | inference procedure，没有进入 inference base | `BaseSpatialInference` 或专用 inference protocol | 中 |

---

# 13. 迁移顺序设计

不能按文件名随意改，也不能一次把 19 个模型全迁。

推荐顺序：

```text
Core contracts
    ↓
GWR canonical migration
    ↓
RGWR / LCRGWR
    ↓
GTWR
    ↓
MGWR
    ↓
MGTWR
    ↓
GWGLM / GWLasso / MixedGWR
    ↓
SGWR / SGTWR
    ↓
LGGWR / GRGWR / ScalableGWR
    ↓
STWR
    ↓
GWPCA / GWDA / GWSS / BootstrapGWR
    ↓
Diagnostics / Plotting legacy adapter removal
    ↓
Public API / dependencies / 0.2.0 cleanup
```

为什么 MGTWR 必须在 MGWR + GTWR 后：

> 它应最终组合“已经稳定的 multiscale capability”和“已经稳定的 spatiotemporal capability”，而不是继续在当前 `MGTWR(MGWR)` 结构上叠加更多技术债。

---

# 14. 详细 PR 路线图

以下是建议实际执行的最小可审查 PR。PR 号仅为预期顺序，真正执行时以 GitHub 分配为准。

---

## Phase A — 架构规则冻结

### A0 — 本文件

**范围**：仅文档。

目标：

- 固化审计结果；
- 固化执行顺序；
- 为其他对话提供唯一 handoff。

验收：

- CI 全绿；
- merge 到 main；
- 本执行台账更新。

---

# Phase B — 建立新的 typed core spine（不改模型行为）

## B1 — 新增 core contracts / types

### 新增建议

```text
src/pygwrx/core/contracts.py
```

先仅加入：

- `DistanceSpec`；
- `LocalSolveResult`；
- `SearchResult`；
- 最小 fitted/view Protocol。

### 这一 PR 禁止

- 不迁移 GWR；
- 不移动旧函数；
- 不改 public constructor；
- 不改数值。

### Tests

新增：

```text
tests/test_core_contracts.py
```

### Typing

将 `contracts.py` 加入 mypy gate。

### Acceptance

- 全部现有测试；
- 50 GWR refs；
- no public behavior change。

---

## B2 — 拆 `utils`: validation + distance compatibility layer

### 新增

```text
core/validation.py
core/distance.py
```

### 迁移

从 `utils.py` 迁出：

- coordinate validation；
- distance metric primitives；
- pairwise distance；
- distance block/row iterators。

### 兼容

第一阶段 `utils.py` 保留 re-export，避免一次破坏 173-symbol API。

例如：

```python
# compatibility import
from .distance import compute_distance_matrix
```

### 特别验证

- Euclidean；
- Manhattan；
- Chebyshev；
- Minkowski p；
- Haversine radius；
- block streaming；
- duplicate coordinates；
- invalid coordinate dimensions。

### GWR Gate

50 reference tests + streaming tests 必须全绿。

---

## B3 — 统一 adaptive bandwidth → distance bandwidth 与 basic weight construction

### 目标

当前 k-th neighbour / compact boundary / duplicate-coordinate 逻辑只有一个 owner。

建议：

```text
core/weights.py
```

提供概念功能：

```text
resolve_local_bandwidth(distances, bandwidth, adaptive)
compute_kernel_weights(...)
```

### 必须统一的语义

- adaptive k 是否包含 self；
- duplicate coordinates；
- k-th neighbour = 0 的 fallback；
- compact kernel 边界 `nextafter`；
- finite/nonnegative weights；
- positive weight count。

### 迁移第一批

只迁：

- GWR bandwidth objective；
- standard GWR weight path；
- core solver compatibility helper。

不要马上迁 19 models。

---

## B4 — 纯化 Gaussian WLS solver

### 目标

`solver.py` 的 canonical primitive 只接收：

```text
X, y, weights
```

返回：

```text
LocalSolveResult
```

### 做法

- 将当前 `_weighted_least_squares_details` 演进为明确 typed canonical internal API；
- public `weighted_least_squares()` 保持兼容 wrapper；
- 清晰区分 unpenalized vs explicit ridge；
- 不让 solver 自己计算 distance / kernel。

### 同时修复

bandwidth objective 的重复 `matrix_rank + SVD`，直接使用 canonical solve.rank。

### Gate

GWR 50 refs 不得发生任何 tolerance 调整。

---

# Phase C — Search 与 GWR bandwidth objective 分离

## C1 — generic search domain

### `optimization.py` 只保留

- exhaustive integer search；
- grid；
- Golden Section；
- Brent；
- caching/evaluation count；
- convergence metadata。

### 移除职责

- coordinate auto bounds；
- GWR-specific bandwidth assumptions。

旧 API 若 public，先 compatibility/deprecation，不直接删除。

---

## C2 — GWR bandwidth objective adapter

标准 GWR CV/AIC/AICc/BIC 作为一个明确的 Gaussian GWR objective 层。

目标：

```text
objective = GWRBandwidthObjective(...)
search = SearchEngine(...)
result = search.minimize(objective)
```

### 必须保持

- full ordered trace；
- invalid candidate = inf；
- adaptive exhaustive semantics；
- boundary_solution provenance；
- current selected minima。

### Gate

- 50 GWR refs；
- bandwidth curve references；
- bandwidth provenance tests。

---

# Phase D — 真正把 BaseSpatialRegressor 变薄

这是全项目重构最关键的一步之一。

## D1 — GWR-specific config 从 Base 移出

从 BaseSpatialRegressor 移走：

```text
kernel
bandwidth
bandwidth_method
adaptive
bandwidth_range
optimization_method
kernel_func_
bandwidth_
GWR local prediction
```

这些应由 GWR 或 capability object 明确拥有。

### BaseSpatialRegressor 最终只保留

- regression fitted state；
- training input/result standard fields；
- score/output contract（若足够通用）；
- lifecycle。

### 非协商条件

`GWR` 的 public constructor 可以保持原样，但参数实际归属改到 GWR 自己。

### 迁移后必须删除的重复

`BaseSpatialRegressor._compute_local_parameters()` 中的具体 GWR algorithm，不能再与 GWR canonical path 并存。

### 测试升级

现有 `test_base_spatial_regressor_hierarchy.py` 不够。

新增 architecture contract tests：

- Base 不拥有 GWR-specific algorithm；
- GWR 拥有 GWR config；
- failed fit state contract；
- training view contract；
- subclass behavior。

---

# Phase E — Fitted/result protocol

## E1 — 标准 fitted-state protocol

目标不是强迫所有模型拥有相同字段，而是建立 capability。

建议区分：

```text
HasCoordinates
HasResponse
HasLocalParameters
HasInference
HasWeights
HasTime
HasDiagnostics
```

### 第一批实现

- GWR；
- GTWR；
- MGWR；
- SGWR；
- ScalableGWR（覆盖 diagnostics 当前常用类型）。

### 保持用户 API

`coef_`, `intercept_`, `diagnostics_` 等用户已经使用的字段不需要为了协议而删除。

协议是标准读取路径，不是强制重新命名所有 public attrs。

---

## E2 — diagnostics 改为 protocol-first

当前：

```text
getattr / aliases first
```

改为：

```text
protocol/view first
legacy alias fallback second
```

迁完全部模型后，再删除 fallback。

### 禁止

新模型迁移后，不允许 diagnostics 再新增该模型的专门 attr alias。

---

## E3 — plotting 只消费 standardized views

逐步移除：

- plotting 中的 distance calculation；
- plotting 中的 condition-number SVD；
- plotting 中的 model-private method 检查。

plotting 仅负责渲染。

---

# Phase F — 模型族迁移

每个模型迁移都必须遵守同一个模板：

1. 先写 architecture contract test；
2. 只移动职责；
3. 不同时“顺便优化算法”；
4. 保持 public outputs；
5. 跑原有 tests；
6. 若有 reference implementation，补/跑 reference；
7. 更新本文件台账。

---

## F1 — GWR

目的：让标准 GWR 成为**新 core 的第一完整消费者**。

必须保持：

- 50 refs；
- no hidden ridge；
- rank policy；
- streaming；
- prediction semantics；
- bandwidth provenance。

同时处理两个现有小债务：

1. GWR docstring 的公式应承认 rank-deficient 情况使用 pseudoinverse/minimum-norm，而不能只写普通 inverse；
2. estimator-level DistanceSpec / distance kwargs 的 API 设计需要完成。

完成后可以正式标记：

> New Core Reference Estimator = GWR

---

## F2 — RGWR + LCRGWR

### RGWR

属于真正 GWR variant，可保留较强的 GWR 关系。

但 protected method 依赖要收敛为明确 capability。

### LCRGWR

必须保留自己的 local compensated ridge 数学；不要强行复用普通 WLS。

目标是共享：

- distance；
- bandwidth resolution；
- kernel；
- common inference assembly；

而 solver 是 LCR-specific。

---

## F3 — GTWR

目标：

- 使用新的 generic SearchResult/search machinery；
- 建立明确的 spatiotemporal distance engine；
- fitted protocol 标准化。

### 不要做

不要在架构迁移 PR 同时把当前 3 个 n×n distance matrices 改成 streaming。

先保持数值，再单独做 memory PR。

---

## F4 — MGWR

这是高风险迁移。

### 当前特性

MGWR backfitting 会反复复用距离，因此：

> **不能机械套用 GWR 的“全部 streaming”策略。**

否则每个 backfitting iteration / variable search 都重复距离计算，CPU 可能严重恶化。

### 目标设计

支持显式 strategy，例如：

```text
auto
cache
stream
```

或 memory-budget-driven policy。

### exact inference

还需要单独处理：

- `partial_R`；
- covariance partial arrays；
- `n_chunks=1` 默认可能造成 O(n²p) 工作内存。

这个属于后续 performance PR，不要和第一次 architecture migration 混在一起。

---

## F5 — MGTWR

必须等 GTWR + MGWR capability 稳定后再做。

目标不一定要求彻底取消 Python 类继承，但概念上应变成：

```text
MGTWR
 = multiscale backfitting capability
 + spatiotemporal distance capability
 + Gaussian local solver
```

而不是：

```text
MGTWR = “因为方便，所以继承整个 MGWR 类，再补时间”
```

任何改变 MGTWR 继承结构的 PR 必须特别关注 public `isinstance` / docs / tests。

---

## F6 — GWGLM / GWLasso / MixedGWR

### GWGLM

长期应脱离 `GWR` 具体类继承。

它应该是：

```text
BaseSpatialRegressor
+ spatial weights/search capability
+ local IWLS
+ family-specific diagnostics
```

### GWLasso

保留 L1 solver；复用 distance/kernel/search。

### MixedGWR

保留 `_mixed_gwr_core` partial regression；重新定义其与公共 core 的边界。

---

## F7 — SGWR / SGTWR

### SGWR

共享：

- spatial distance；
- basic geographic kernel；
- generic search；
- fitted/result protocol。

保留 model-specific：

- similarity kernel；
- alpha mixing；
- attribute-space logic。

### SGTWR

在 GTWR capability 上增加 similarity composer，而不是重新复制 GTWR 核心。

---

## F8 — LGGWR / GRGWR / ScalableGWR

### LGGWR

不要强制使用普通 geographic distance backend。

应把 learned geometry 作为 custom `DistanceStrategy` / geometry provider。

### GRGWR

GWR initial surface 用 composition；regime clustering / ICM 保持模型内。

### ScalableGWR

其 KDTree/compressed moment engine 是核心创新/效率来源，**禁止为了“统一”改成 dense/streamed ordinary GWR**。

只统一：

- lifecycle；
- result protocol；
- diagnostics/plotting interface。

---

## F9 — STWR

STWR 使用 stage-based input，与普通 `fit(X,y,coords)` 不是完全同一 contract。

因此不要为了继承形式强行破坏其 API。

目标是：

- 进入统一 fitted lifecycle；
- 提供 standardized output views；
- stage/time capability 明确。

---

## F10 — GWPCA / GWDA / GWSS / BootstrapGWR

这是验证 base category 是否真正有用的一步。

### GWPCA

迁入：

```text
BaseSpatialTransformer
```

### GWDA

迁入：

```text
BaseSpatialClassifier
```

### GWSS

迁入：

```text
BaseSpatialStatistics
```

### BootstrapGWR

迁入：

```text
BaseSpatialInference
```

或在实施前若发现 `BaseSpatialInference` 语义过窄，可设计更合适的 inference procedure contract。

这里不能为了“让 base 有人继承”而破坏模型自然 API；如果 base 设计不合适，应修改 base。

---

# Phase G — 下游收口

当 19 个模型都实现新 protocol 后：

## G1 diagnostics cleanup

删除：

- 大量 `first_available` alias；
- migrated models 的 model-specific `getattr`；
- 重复 weight reconstruction。

保留明确 model-specific diagnostics API。

## G2 plotting cleanup

删除计算逻辑；仅使用 standardized view。

## G3 io cleanup

将 core 中残留 GeoPandas / file concerns 迁至 io。

---

# Phase H — Public API 与 0.2.0 架构版

最后再处理 breaking/public surface。

## H1 — 173-symbol public API 审计

分类：

```text
Tier 1: end-user stable
Tier 2: advanced public
Tier 3: internal/private
```

重点审查：

- mixins 是否应该 public；
- low-level solver 是否应该作为普通用户 API；
- `DistanceCache`；
- `local_regression`；
- optimizer classes；
- compatibility aliases。

## H2 — dependency policy

再决定 matplotlib/geopandas/shapely 是否继续 mandatory。

## H3 — deprecated compatibility removal

在架构稳定后再删旧 re-export / legacy adapters。

## H4 — 发布

建议把整个架构收敛作为：

```text
pyGWRx 0.2.0
```

而不是悄悄塞进 0.1.x patch。

---

# 15. Architecture Contract Tests 设计

除了模型数值测试，需要新增一类“架构回归测试”。

建议未来创建：

```text
tests/architecture/
```

或保持当前 flat tests，但统一命名：

```text
test_architecture_*.py
```

至少检查：

## 15.1 estimator lifecycle

每个 estimator：

- fit 前 unfitted；
- fit 成功后 fitted；
- failed refit 不保留伪旧状态；
- feature names contract；
- n_samples/n_features contract。

## 15.2 capability declarations

明确哪些模型：

- predict；
- transform；
- predict_proba；
- time；
- local parameters；
- inference；
- stored weights。

## 15.3 core ownership

迁移完成后，可用 AST/static test 禁止：

- migrated plain-Gaussian model 直接新写 `np.linalg.*` WLS；
- plotting 直接计算 distance matrix；
- diagnostics 对 migrated model 增加 attr aliases；
-多个模块重新实现 adaptive k boundary semantics。

## 15.4 result consistency

例如：

- `to_frame()` rows = fitted locations；
- parameter names/order stable；
- coords order stable；
- prediction result names stable。

---

# 16. 每一个实际重构 PR 的统一验收模板

未来每个 PR description 建议复制以下 checklist：

```text
[ ] 只包含本阶段职责，不混入无关算法增强
[ ] Black
[ ] isort
[ ] Ruff
[ ] mypy
[ ] non-reference tests
[ ] cross-platform CI
[ ] minimum dependency tests
[ ] coverage threshold
[ ] generated docs clean
[ ] build wheel/sdist
[ ] security/SBOM
[ ] GWR 50 external-reference tests（若 core/GWR 受影响则必须）
[ ] 目标模型 reference tests（若存在）
[ ] API/output shape 未意外变化
[ ] failed-fit state contract 未破坏
[ ] memory behavior 未意外退化
[ ] 本 master plan 执行台账已更新
```

如果 PR 涉及 hot numerical path，再增加：

```text
[ ] before/after runtime benchmark
[ ] before/after peak memory benchmark
[ ] no new accidental O(n²) allocation
```

---

# 17. 执行台账（后续对话必须维护）

状态定义：

- `DONE`：已 merge 到 main，且验证完成；
- `IN PROGRESS`：已有 PR/branch；
- `PENDING`：尚未开始；
- `BLOCKED`：发现需要先解决的问题。

| Phase | 工作 | 状态 | PR | Merge SHA | 验证/备注 |
|---|---|---|---|---|---|
| A0 | 全项目架构审计 + 本 master plan | IN PROGRESS | 待创建 | — | 基线 `241246b9...` |
| B1 | typed core contracts | PENDING | — | — | 第一个代码重构 PR |
| B2 | split validation/distance from utils | PENDING | — | — | compatibility re-export |
| B3 | canonical bandwidth→weight semantics | PENDING | — | — | duplicate/tie/boundary tests |
| B4 | pure rank-aware WLS result contract | PENDING | — | — | 移除 bandwidth double decomposition |
| C1 | generic search engine cleanup | PENDING | — | — | optimization 不再知道 GWR |
| C2 | GWR bandwidth objective adapter | PENDING | — | — | full trace 必须不变 |
| D1 | thin BaseSpatialRegressor | PENDING | — | — | 最高优先结构 PR |
| E1 | fitted/capability protocol | PENDING | — | — | protocol-first |
| E2 | diagnostics protocol-first | PENDING | — | — | legacy fallback 暂留 |
| E3 | plotting view-only | PENDING | — | — | 删除统计重算 |
| F1 | GWR canonical new-core migration | PENDING | — | — | 50 refs |
| F2 | RGWR/LCRGWR migration | PENDING | — | — | — |
| F3 | GTWR migration | PENDING | — | — | — |
| F4 | MGWR migration | PENDING | — | — | 不可盲目 streaming |
| F5 | MGTWR migration | PENDING | — | — | 最高风险 |
| F6 | GWGLM/GWLasso/MixedGWR | PENDING | — | — | — |
| F7 | SGWR/SGTWR | PENDING | — | — | — |
| F8 | LGGWR/GRGWR/ScalableGWR | PENDING | — | — | specialized engines 保留 |
| F9 | STWR | PENDING | — | — | staged protocol |
| F10 | GWPCA/GWDA/GWSS/BootstrapGWR | PENDING | — | — | 验证 base categories |
| G1 | diagnostics legacy cleanup | PENDING | — | — | 所有模型迁移后 |
| G2 | plotting cleanup | PENDING | — | — | — |
| G3 | IO/core boundary cleanup | PENDING | — | — | — |
| H1 | public API tier audit | PENDING | — | — | 173 symbols |
| H2 | dependency policy | PENDING | — | — | 后期单独决策 |
| H3 | legacy/deprecation cleanup | PENDING | — | — | — |
| H4 | 0.2.0 architecture release | PENDING | — | — | full release gates |

---

# 18. 下一项实际工作：B1

当本文件合并后，**下一对话不要重新讨论先做什么**。

直接执行：

> **B1 — Introduce typed core architecture contracts without changing model behavior.**

建议 branch：

```text
refactor/core-contracts
```

建议 PR 内容严格限制为：

```text
src/pygwrx/core/contracts.py
src/pygwrx/core/__init__.py       # 只有确实需要 public/advanced export 时才改
tests/test_core_contracts.py
pyproject.toml                    # 加入 mypy 文件范围
docs/...                          # 仅在 public surface 变化时生成
ARCHITECTURE_REFACTOR_MASTER_PLAN.md
```

第一版 contract 不要设计得过度复杂。

只建立后续所有重构都确定会需要的最小对象：

1. `DistanceSpec`；
2. `LocalSolveResult`；
3. `SearchResult`；
4. 极少量 fitted/view Protocol。

不要在 B1 同时移动 `utils.py` 或 GWR。

---

# 19. 明确暂缓的问题

以下问题已经识别，但不是第一批 PR：

1. MGWR dense distance strategy；
2. MGWR exact inference O(n²p) memory；
3. GTWR 三套 dense n×n matrices；
4. MGTWR memory；
5. optional dependency 重组；
6. public API 大规模删除；
7. model handbook 全面重写；
8. performance benchmark 全项目统一；
9. 并行/Numba/GPU；
10. 新模型功能。

必须先完成 core architecture spine。

---

# 20. 重构期间的禁止事项

## 不允许 big-bang

不要创建一个 PR 同时修改：

```text
base + core + 19 models + diagnostics + plotting
```

这种 PR 无法可靠定位数值漂移。

## 不允许为了统一破坏模型数学

以下算法必须保留自身专用实现：

- GWGLM IWLS；
- GWLasso L1；
- LCR compensated ridge；
- MGWR/MGTWR backfitting；
- STWR staged temporal formulation；
- SGWR similarity formulation；
- LGGWR learned geometry；
- GRGWR regime optimization；
- ScalableGWR compressed/KDTree engine。

统一的是 infrastructure，不是把所有模型变成 GWR。

## 不允许“全部 streaming”教条

GWR 的 streaming 是正确的，但 MGWR/MGTWR 反复复用距离，可能更适合 cache/memory-budget strategy。

## 不允许 hidden ridge 回归

标准 Gaussian WLS 仍保持 unpenalized default。任何 regularization 必须显式属于具体模型/solver。

## 不允许 diagnostics/plotting 成为第二套模型引擎

它们不能自己重建一整套 model weighting/solver，只能消费标准能力或明确的 model-specific diagnostics service。

---

# 21. 最终完成标准

整个 architecture refactor 只有达到下面状态才算真正结束：

### Core

- 每个 core 模块职责单一；
- distance / weights / search / solver / diagnostics ownership 唯一；
- 没有关键算法的多份漂移实现。

### Models

- 19 个 estimator 都进入明确 lifecycle/capability contract；
- 不再靠“方便复用”形成明显错误的概念继承；
- model-specific math 仍然清楚可读。

### Diagnostics

- 不再需要大规模 attr alias guessing；
- common views typed；
- model-specific diagnostics 明确标记。

### Plotting

- 不重新计算模型；
- 只读取 standardized views。

### Tests

- architecture contract tests 完整；
- GWR 50 refs 保留；
- 非 GWR 关键模型逐步补 external references；
- cross-platform CI 全绿。

### Typing

- core architectural spine 全部 mypy；
- migrated models 逐步 typed；
- 不再依靠大量 `Any` 维持内部一致性。

### Public API

- 清楚区分 user / advanced / internal；
- 0.2.0 有明确迁移说明；
- 顶层 `pygwrx.GWR` 等核心用户入口尽量保持稳定。

---

# 22. 最终目标图

```text
                         ┌─────────────────────┐
                         │   User-facing API   │
                         │ GWR/MGWR/GTWR/...   │
                         └──────────┬──────────┘
                                    │
                    thin estimator lifecycle/contracts
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
      ┌───────▼────────┐                         ┌────────▼────────┐
      │  Model math    │                         │ Standard views  │
      │ backfit/IWLS/  │                         │ result/diag/etc │
      │ similarity/... │                         └────────┬────────┘
      └───────┬────────┘                                  │
              │                                  ┌────────▼────────┐
              │                                  │ diagnostics     │
              │                                  │ plotting        │
              │                                  │ io/export       │
              │                                  └─────────────────┘
              │
      ┌───────▼───────────────────────────────────────────────┐
      │                    Numerical Core                    │
      │ distance | kernels | weights | search | solver       │
      │ metrics  | Gaussian diagnostics | typed contracts    │
      └───────────────────────────────────────────────────────┘
```

这才是 pyGWRx 下一阶段应达到的软件形态：

> **不是“19 个模型各写一套再用 adapter 拼起来”，而是“一个稳定数值核心 + 明确的模型数学层 + 标准结果协议 + 薄的展示与 I/O 层”。**

---

# 23. 审计后的总体判断

当前 pyGWRx 的问题并不是“代码质量低”，相反，测试、文档、CI、模型数量和标准 GWR 数值验证已经很强。

真正的问题是项目发展速度较快后形成的**架构成熟度滞后于功能成熟度**：

- 功能已经像一个成熟软件包；
- 内部仍保留多个历史阶段的设计模式。

这也是现在进行大重构的最佳窗口，因为项目仍处于 Alpha，且标准 GWR 已经有强 reference suite 可以作为安全锚点。

因此下一阶段的核心目标不再是“继续加更多功能”，而是：

> **把已经存在的功能重新组织成一个可长期维护、可扩展、可验证、适合软件论文与后续模型研发的统一架构。**

本文件应一直保留到 0.2.0 架构版完成，并在每个 PR 后持续更新执行台账。
