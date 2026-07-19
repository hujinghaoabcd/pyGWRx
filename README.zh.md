<p align="center">
  <img src="./docs/assets/images/logo.svg" alt="pyGWRx" width="460">
</p>
<p align="center">
  面向地理加权回归、局部空间统计、时空建模、诊断与可视化的 Python 研究型库
</p>

<p align="center">
  <a href="https://github.com/hujinghaoabcd/pyGWRx/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-139C5A.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11--3.14-174D5B.svg">
  <img alt="Status" src="https://img.shields.io/badge/Status-Alpha-F4B942.svg">
  <img alt="Models" src="https://img.shields.io/badge/Public_models-19-139C5A.svg">
  <img alt="Public API examples" src="https://img.shields.io/badge/Public_API_examples-174%2F174-087F5B.svg">
  <img alt="Examples" src="https://img.shields.io/badge/Runnable_examples-45-2F9E72.svg">
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/zh/">中文文档</a> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/zh/models/">中文模型手册</a> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/examples/">示例</a> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/api/">API</a>
</p>

---

## pyGWRx 是什么

pyGWRx 是一个面向**地理加权建模**的 Python 实现与研究平台。项目在统一数值底座上提供经典 GWR、多尺度与稳健扩展、广义响应、时空邻域、局部正则化、多变量分析、可扩展近似、属性相似性加权以及原创研究模型。

项目分为五层：

1. **模型层**：19 个正式公开模型。
2. **核心数值层**：核函数、距离、局部求解器、带宽选择、优化、指标、输入验证和基类。
3. **诊断层**：模型摘要、残差、影响度、参数推断、局部共线性、时间、权重和空间机制诊断。
4. **可视化层**：56 个模型感知或数组输入的 Matplotlib 绘图函数。
5. **I/O 与示例层**：统一支持 NumPy、pandas、GeoPandas 与 Shapely，并提供 45 个隔离运行示例。

> pyGWRx 采用统一的 **fit → 结果检查 → 诊断 → 可视化** 使用方式，并明确不实现 scikit-learn estimator contract，也不承诺 `Pipeline`、`GridSearchCV`、`clone` 或 `check_estimator` 兼容。

## 主要特点

- **19 个模型集中管理**：经典、多尺度、稳健、广义、时空、正则化、多变量、可扩展、相似性和原创研究模型。
- **能力边界明确**：回归、分类、变换、局部统计和推断模型不会被笼统描述成同一种“预测器”。
- **详细模型手册**：每个模型都有原理、公式、算法步骤、参数、结果、诊断、限制、报告要求、图件和完整代码。
- **174/174 API—示例覆盖**：每个公开接口都映射到正式示例和自动生成的 API 页面。
- **完整空间分析基础安装**：默认同时安装 NumPy、SciPy、pandas、Matplotlib、GeoPandas 和 Shapely，安装后即可使用地图与 GeoDataFrame 工作流。
- **可复现研究流程**：确定性示例数据、显式随机种子、严格文档构建和可选参考实现对照测试。

## 安装

支持 **Python 3.11–3.14**。Alpha 阶段建议从源码安装：

```bash
git clone https://github.com/hujinghaoabcd/pyGWRx.git
cd pyGWRx
python -m pip install --upgrade pip
python -m pip install -e .
```

Matplotlib、GeoPandas 和 Shapely 已包含在普通安装中。其余可选功能按需安装：

```bash
python -m pip install -e ".[ml]"        # GWLasso、GWPCA、GRGWR
python -m pip install -e ".[parquet]"   # PyArrow
python -m pip install -e ".[all]"       # 其余全部用户功能
python -m pip install -e ".[test]"      # 测试环境
python -m pip install -e ".[dev]"       # 开发与构建工具
python -m pip install -e ".[docs]"      # 文档工具链
python -m pip install -e ".[reference]" # 可选参考实现对照测试
```

基础安装已经包含 Matplotlib、GeoPandas 和 Shapely；scikit-learn 与 PyArrow 仍作为可选依赖。

## 五分钟 GWR 示例

```python
import numpy as np
import pandas as pd
from pygwrx import GWR

rng = np.random.default_rng(42)
n = 80
coords = pd.DataFrame(rng.uniform(0, 10, size=(n, 2)), columns=["east", "north"])
X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["income", "access"])
local_income = 1.0 + 0.15 * coords["east"].to_numpy()
y = 2.0 + local_income * X["income"] - 0.7 * X["access"]
y += rng.normal(scale=0.35, size=n)

model = GWR(kernel="bisquare", bandwidth=28, adaptive=True)
model.fit(X, y, coords)

print(model.summary())
print(model.to_frame().head())
print("R²:", model.score(X, y, coords))

result = model.predict_result(X.iloc[:4], coords.iloc[:4])
print(result.to_frame())
```

### 诊断与绘图

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame
from pygwrx.plotting import plot_coefficient_map, plot_diagnostic_panel

print(diagnostics_frame([model], labels=["GWR"]))
print(local_diagnostic_frame(model).head())

fig, ax = plot_coefficient_map(model, feature="income", theme="paper")
fig.savefig("income_coefficient.png", dpi=200, bbox_inches="tight")

fig, axes = plot_diagnostic_panel(model, theme="paper")
fig.savefig("gwr_diagnostics.png", dpi=200, bbox_inches="tight")
```

所有绘图函数返回 Matplotlib 对象，不会自动调用 `plt.show()`。

## 模型目录与能力矩阵

| 模型 | 功能 | 输入 | 新位置操作 | Extra | 示例 |
|---|---|---|---|---|---|
| [`GWR`](docs/models/gwr.md) | Classic local regression | X, y, coords | predict / predict_result | `base` | [代码](examples/models/01_gwr.py) |
| [`MGWR`](docs/models/mgwr.md) | Variable-specific spatial scales | X, y, coords | calibration only | `base` | [代码](examples/models/02_mgwr.py) |
| [`RGWR`](docs/models/rgwr.md) | Outlier-resistant local regression | X, y, coords | predict / predict_result | `base` | [代码](examples/models/03_rgwr.py) |
| [`STWR`](docs/models/stwr.md) | Stage-based spatiotemporal regression | stage lists + intervals | predict / predict_result | `base` | [代码](examples/models/04_stwr.py) |
| [`GTWR`](docs/models/gtwr.md) | Row-wise space-time regression | X, y, coords, times | predict / predict_result | `base` | [代码](examples/models/05_gtwr.py) |
| [`GWGLM`](docs/models/gwglm.md) | Gaussian, binomial, Poisson local GLM | X, y, coords (+ exposure) | predict / predict_result | `base` | [代码](examples/models/06_gwglm.py) |
| [`GWLasso`](docs/models/gw-lasso.md) | Locally sparse regression | X, y, coords | predict | `ml` | [代码](examples/models/07_gw_lasso.py) |
| [`MixedGWR`](docs/models/mixed-gwr.md) | Global + local coefficients | X, y, coords + variable sets | predict | `base` | [代码](examples/models/08_mixed_gwr.py) |
| [`GWPCA`](docs/models/gwpca.md) | Local principal components | X, coords | transform | `ml` | [代码](examples/models/09_gwpca.py) |
| [`GWDA`](docs/models/gwda.md) | Local discriminant classification | X, labels, coords | predict / predict_proba | `base` | [代码](examples/models/10_gwda.py) |
| [`GWSS`](docs/models/gwss.md) | Local descriptive statistics | X, coords | statistics only | `base` | [代码](examples/models/11_gwss.py) |
| [`ScalableGWR`](docs/models/scalable-gwr.md) | Polynomial-kernel approximation | X, y, coords | predict / predict_result | `base` | [代码](examples/models/12_scalable_gwr.py) |
| [`LCRGWR`](docs/models/lcr-gwr.md) | Local ridge compensation | X, y, coords | predict / predict_result | `base` | [代码](examples/models/13_lcr_gwr.py) |
| [`BootstrapGWR`](docs/models/bootstrap-gwr.md) | Non-stationarity inference | X, y, coords | inference only | `base` | [代码](examples/models/14_bootstrap_gwr.py) |
| [`SGWR`](docs/models/sgwr.md) | Geography + attribute similarity | X, y, coords + similarity vars | predict / predict_result | `base` | [代码](examples/models/15_sgwr.py) |
| [`SGTWR`](docs/models/sgtwr.md) | Space + time + similarity | X, y, coords, times + similarity vars | predict / predict_result | `base` | [代码](examples/models/16_sgtwr.py) |
| [`MGTWR`](docs/models/mgtwr.md) | Variable-specific space-time scales | X, y, coords, times | calibration only | `base` | [代码](examples/models/17_mgtwr.py) |
| [`LGGWR`](docs/models/lg-gwr.md) | Learned latent neighbourhood geometry | X, y, coords, attributes | predict / predict_result | `base` | [代码](examples/models/18_lg_gwr.py) |
| [`GRGWR`](docs/models/gr-gwr.md) | Connected spatial regimes | X, y, coords | predict / predict_result | `ml` | [代码](examples/models/19_gr_gwr.py) |

### 需要特别注意的能力边界

- `MGWR` 和 `MGTWR` 当前只提供校准位置结果，会明确拒绝尚未验证的独立新位置预测。
- `GWPCA` 是局部变换模型，使用 `transform()`。
- `GWSS` 计算局部描述统计。
- `BootstrapGWR` 检验系数空间非平稳性，不负责响应预测。
- `GWDA` 是分类模型，提供 `predict()` 与 `predict_proba()`。
- `MGTWR` 已在 pyGWRx 内部完整实现，不需要模型专用外部运行依赖。
- `mgwr` 与 `spglm` 只用于可选的 GWGLM 参考对照测试，正常 GWGLM 拟合不会调用这两个包。
- `LGGWR` 和 `GRGWR` 是原创研究模型，应报告初始化、敏感性、收敛与验证范围。

## 如何选择模型

| 研究需求 | 建议起点 | 有充分依据后再增加 |
|---|---|---|
| 连续响应、平滑空间变化 | `GWR` | `MGWR`、`RGWR`、`LCRGWR`、`ScalableGWR` |
| 二元或计数响应 | 全局 GLM + `GWGLM` | 家族特定局部诊断 |
| 空间与逐行时间 | `GTWR` | `SGTWR`、`MGTWR` |
| 分阶段历史过程 | `STWR` | 阶段与参数敏感性分析 |
| 全局效应与局部效应并存 | 全局回归 + `MixedGWR` | 有理论依据的变量划分 |
| 局部变量选择 | `GWLasso` | 稳定性和重采样分析 |
| 局部多变量结构 | `GWSS`、`GWPCA` | `GWDA` 局部分类 |
| 地理邻近 + 功能相似 | `SGWR` | `SGTWR` 或研究型 `LGGWR` |
| 连续空间机制分区 | 标准 GWR | 研究型 `GRGWR` |

完整说明见[模型选择指南](https://hujinghaoabcd.github.io/pyGWRx/zh/guides/model-selection/)和[19 个模型中文手册](https://hujinghaoabcd.github.io/pyGWRx/zh/models/)。

## 核函数、带宽与距离

内置核函数：

```python
from pygwrx.core import (
    gaussian_kernel,
    bisquare_kernel,
    exponential_kernel,
    tricube_kernel,
    boxcar_kernel,
)
```

- **固定带宽**：坐标距离单位下的距离。
- **自适应带宽**：整数近邻数，不同位置对应的实际距离不同。
- **紧支撑核**：bisquare、tricube、boxcar 在带宽之外权重严格为零。
- **连续核**：Gaussian、exponential 随距离连续衰减。

论文或报告中应说明坐标参考系、距离度量、核函数、固定/自适应模式、选择准则、搜索范围和最终带宽。多尺度模型需要报告每个参数的尺度。

## 全部公开功能与示例

项目共有 **174 个公开 API**：

| 命名空间 | 内容 |
|---|---|
| `pygwrx.models` | 模型与类型化预测结果对象 |
| `pygwrx.core` | 核、距离、带宽、优化、求解器、指标与基类 |
| `pygwrx.diagnostics` | 模型、残差、影响、推断、共线性、时间、权重与机制诊断 |
| `pygwrx.plotting` | 系数、残差、比较、时空、权重分解、潜在几何与机制绘图 |
| `pygwrx.io` | 数据转换、持久化和数据集注册表 |

每个公开接口都具有：

- 自动生成的签名和完整 Docstring；
- 用途摘要与导入路径；
- 对应正式示例链接；
- API 页面中内嵌的完整示例源码；
- `examples/API_COVERAGE.json` 和 `.csv` 中的映射记录。

检查接口—示例契约：

```bash
python tools/generate_api_docs.py
python tools/generate_example_docs.py
python examples/validate_coverage.py
```

运行全部示例：

```bash
python -m pip install -e ".[all,test]"
python examples/run_all.py
```

示例构成：

- 19 个模型示例
- 8 个核心数值示例
- 5 个诊断示例
- 6 个绘图示例
- 4 个 I/O 示例
- 3 个完整工作流

## 文档体系

MkDocs 站点包括：

- 入门说明、数据契约与模型选择
- 19 个模型的英文详细指南
- 19 个模型的中文详细指南
- Core、Diagnostics、Plotting、I/O 功能手册
- 45 个示例的完整源码
- 47 张图件组成的可视化图谱
- 174 个公开接口的生成式 API 页面
- 完整算法百科与两个原创模型专论
- 测试、贡献、发布、引用和 API 稳定性说明

本地启动：

```bash
python -m pip install -e ".[docs]"
python tools/generate_api_docs.py
python tools/generate_example_docs.py
mkdocs serve
# 浏览器打开 http://127.0.0.1:8000

mkdocs build --strict --clean
```

## 验证状态

版本 **0.1.2** 定位为 **Alpha 研究版本**。正式公开模型已经进入统一文档、示例和测试体系，但不同模型的能力并不完全相同，0.x 阶段 API 仍可能调整。

当前稳定测试基线：

```text
363 passed
```

该 Linux/Python 3.13 本地结果包含 360 个非 reference 测试（其中包括不依赖外部包的 MGTWR 冻结数值基准）以及 3 个通过可选 `reference` 分组执行的独立 GWGLM 数值对照测试。Windows/Linux/macOS × Python 3.11–3.14 的 12 组合矩阵已作为阻断式 GitHub Actions 工作流配置；在对应远程运行成功前，不将其表述为已通过。

## 数据与法律提示

开发树中含有第三方示例数据。文献引用、允许学术使用和允许将数据复制进 Python wheel 再分发是三件不同的事情。正式公开分发前必须逐项核查原始许可证和署名要求。

MIT 许可证覆盖 pyGWRx 自有源码，不会自动重新许可第三方数据或依赖。

## 项目结构

```text
pyGWRx/
├── src/pygwrx/          # 正式代码包
├── tests/               # 稳定测试套件
├── examples/            # 45 个示例和 API 覆盖清单
├── docs/                # 中英文手册、API、理论说明与图谱
├── tools/               # API/文档生成工具
├── pyproject.toml       # 包元数据与可选依赖
├── mkdocs.yml           # 文档布局
├── README.md            # 英文首页
└── README.zh.md         # 中文首页
```

历史原型、旧示例、旧文档、旧 GTWR 报告接口和非发行审计资料均保存在单独归档中，不再混入正式项目树。

## 作者、引用与许可证

- 作者：**Jinghao Hu**
- 引用信息：[`CITATION.cff`](CITATION.cff)
- 源码许可证：[MIT](LICENSE)
- 文档：<https://hujinghaoabcd.github.io/pyGWRx/>


## 真实数据五分钟入门

```python
from pygwrx import GWR
from pygwrx.io import load_columbus

data = load_columbus(return_type="dict")
model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(
    data["data"], data["target"], data["coords"]
)
print(model.summary())
```

pyGWRx 采用面向空间建模任务的接口，不承诺 scikit-learn 的克隆、流水线、
`get_params()` 或 `set_params()` 协议。内置数据集保留上游许可证，详见
`DATA_LICENSES.md`。
