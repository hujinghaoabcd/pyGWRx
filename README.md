<p align="center">
  <img src="https://raw.githubusercontent.com/hujinghaoabcd/pyGWRx/main/docs/assets/images/logo.svg" alt="pyGWRx" width="460">
</p>

<p align="center">
  A research-oriented Python library for geographically weighted regression,
  local spatial statistics, spatiotemporal modelling, diagnostics, and visualization.
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
  <b>English</b> · <a href="README.zh.md">简体中文</a> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/">Documentation</a> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/models/">Model Handbook</a> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/examples/">Examples</a> ·
  <a href="https://hujinghaoabcd.github.io/pyGWRx/api/">API</a>
</p>

---

## What pyGWRx is

pyGWRx is a Python implementation and research platform for **geographically weighted modelling**. It provides a common numerical foundation for classic GWR, multiscale and robust extensions, generalized responses, spatiotemporal neighbourhoods, local regularization, multivariate methods, scalable approximations, similarity-based weighting, and original research models.

The library is designed around five layers:

1. **Models** — 19 supported public model classes.
2. **Core numerics** — kernels, distances, local solvers, bandwidth selection, optimization, metrics, validation, and base classes.
3. **Diagnostics** — model summaries, residuals, influence, parameter inference, local collinearity, time, weight, and regime diagnostics.
4. **Visualization** — 56 model-aware and array-based Matplotlib functions.
5. **I/O and examples** — NumPy, pandas, GeoPandas, and Shapely data contracts plus 45 isolated runnable scripts.

> pyGWRx follows a consistent **fit → inspect → diagnose → visualize** style. It deliberately does **not** implement the scikit-learn estimator contract, `Pipeline`, `GridSearchCV`, `clone`, or `check_estimator`.

## Why use it

- **One documented model family:** classic, multiscale, robust, generalized, temporal, regularized, multivariate, scalable, similarity-based, and research models in one package.
- **Explicit capability boundaries:** regression, classification, transformation, local statistics, and inference models are not presented as interchangeable predictors.
- **Complete documentation:** every model page includes theory, equations, fitting steps, parameters, outputs, diagnostics, limitations, reporting guidance, figures, and a full runnable example.
- **174/174 API-to-example coverage:** every public symbol is mapped to a maintained script and generated API page.
- **Complete spatial base install:** NumPy, SciPy, pandas, Matplotlib, GeoPandas, and Shapely are installed together so mapping and GeoDataFrame workflows work immediately.
- **Research reproducibility:** deterministic example data, explicit random seeds, strict documentation builds, and reference-comparison tests where available.

## Installation

pyGWRx supports **Python 3.11–3.14**. During the Alpha phase, install from a source checkout:

```bash
git clone https://github.com/hujinghaoabcd/pyGWRx.git
cd pyGWRx
python -m pip install --upgrade pip
python -m pip install -e .
```

Matplotlib, GeoPandas, and Shapely are included in the normal installation. Add only the remaining optional features you need:

```bash
python -m pip install -e ".[ml]"        # GWLasso, GWPCA, GRGWR
python -m pip install -e ".[parquet]"   # PyArrow persistence
python -m pip install -e ".[all]"       # all remaining user-facing extras
python -m pip install -e ".[test]"      # tests
python -m pip install -e ".[dev]"       # development/build tooling
python -m pip install -e ".[docs]"      # MkDocs documentation toolchain
python -m pip install -e ".[reference]" # optional numerical-reference tests
```

The base installation includes Matplotlib, GeoPandas, and Shapely. scikit-learn and PyArrow remain optional.

## Five-minute GWR example

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

### Diagnostics and plotting

```python
from pygwrx.diagnostics import (
    diagnostics_frame,
    local_diagnostic_frame,
    parameter_significance,
)
from pygwrx.plotting import plot_coefficient_map, plot_diagnostic_panel

print(diagnostics_frame([model], labels=["GWR"]))
print(local_diagnostic_frame(model).head())
print(parameter_significance(model, alpha=0.05, correction="fdr_bh").head())

fig, ax = plot_coefficient_map(model, feature="income", theme="paper")
fig.savefig("income_coefficient.png", dpi=200, bbox_inches="tight")

fig, axes = plot_diagnostic_panel(model, theme="paper")
fig.savefig("gwr_diagnostics.png", dpi=200, bbox_inches="tight")
```

Plotting functions return Matplotlib objects and never call `plt.show()` automatically.

## Model catalogue and capability matrix

| Model | Purpose | Required inputs | New-location operation | Extra | Example |
|---|---|---|---|---|---|
| [`GWR`](docs/models/gwr.md) | Classic local regression | X, y, coords | predict / predict_result | `base` | [code](examples/models/01_gwr.py) |
| [`MGWR`](docs/models/mgwr.md) | Variable-specific spatial scales | X, y, coords | calibration only | `base` | [code](examples/models/02_mgwr.py) |
| [`RGWR`](docs/models/rgwr.md) | Outlier-resistant local regression | X, y, coords | predict / predict_result | `base` | [code](examples/models/03_rgwr.py) |
| [`STWR`](docs/models/stwr.md) | Stage-based spatiotemporal regression | stage lists + intervals | predict / predict_result | `base` | [code](examples/models/04_stwr.py) |
| [`GTWR`](docs/models/gtwr.md) | Row-wise space-time regression | X, y, coords, times | predict / predict_result | `base` | [code](examples/models/05_gtwr.py) |
| [`GWGLM`](docs/models/gwglm.md) | Gaussian, binomial, Poisson local GLM | X, y, coords (+ exposure) | predict / predict_result | `base` | [code](examples/models/06_gwglm.py) |
| [`GWLasso`](docs/models/gw-lasso.md) | Locally sparse regression | X, y, coords | predict | `ml` | [code](examples/models/07_gw_lasso.py) |
| [`MixedGWR`](docs/models/mixed-gwr.md) | Global + local coefficients | X, y, coords + variable sets | predict | `base` | [code](examples/models/08_mixed_gwr.py) |
| [`GWPCA`](docs/models/gwpca.md) | Local principal components | X, coords | transform | `ml` | [code](examples/models/09_gwpca.py) |
| [`GWDA`](docs/models/gwda.md) | Local discriminant classification | X, labels, coords | predict / predict_proba | `base` | [code](examples/models/10_gwda.py) |
| [`GWSS`](docs/models/gwss.md) | Local descriptive statistics | X, coords | statistics only | `base` | [code](examples/models/11_gwss.py) |
| [`ScalableGWR`](docs/models/scalable-gwr.md) | Polynomial-kernel approximation | X, y, coords | predict / predict_result | `base` | [code](examples/models/12_scalable_gwr.py) |
| [`LCRGWR`](docs/models/lcr-gwr.md) | Local ridge compensation | X, y, coords | predict / predict_result | `base` | [code](examples/models/13_lcr_gwr.py) |
| [`BootstrapGWR`](docs/models/bootstrap-gwr.md) | Non-stationarity inference | X, y, coords | inference only | `base` | [code](examples/models/14_bootstrap_gwr.py) |
| [`SGWR`](docs/models/sgwr.md) | Geography + attribute similarity | X, y, coords + similarity vars | predict / predict_result | `base` | [code](examples/models/15_sgwr.py) |
| [`SGTWR`](docs/models/sgtwr.md) | Space + time + similarity | X, y, coords, times + similarity vars | predict / predict_result | `base` | [code](examples/models/16_sgtwr.py) |
| [`MGTWR`](docs/models/mgtwr.md) | Variable-specific space-time scales | X, y, coords, times | calibration only | `base` | [code](examples/models/17_mgtwr.py) |
| [`LGGWR`](docs/models/lg-gwr.md) | Learned latent neighbourhood geometry | X, y, coords, attributes | predict / predict_result | `base` | [code](examples/models/18_lg_gwr.py) |
| [`GRGWR`](docs/models/gr-gwr.md) | Connected spatial regimes | X, y, coords | predict / predict_result | `ml` | [code](examples/models/19_gr_gwr.py) |

### Important boundaries

- `MGWR` and `MGTWR` provide calibration-location results but intentionally reject unvalidated independent-target prediction.
- `GWPCA` is a local transformer and uses `transform()`.
- `GWSS` computes local descriptive statistics.
- `BootstrapGWR` performs coefficient non-stationarity inference rather than response prediction.
- `GWDA` is a classifier and provides `predict()`/`predict_proba()`.
- `MGTWR` is implemented entirely inside pyGWRx and has no model-specific runtime dependency.
- `mgwr` and `spglm` are used only in optional GWGLM reference-comparison tests, not during ordinary GWGLM fitting.
- `LGGWR` and `GRGWR` are original research models; report sensitivity, initialization, convergence, and validation scope.

## Choosing a model

| Scientific need | Start with | Add only when justified |
|---|---|---|
| Continuous response, smooth spatial variation | `GWR` | `MGWR`, `RGWR`, `LCRGWR`, `ScalableGWR` |
| Binary or count response | global GLM + `GWGLM` | family-specific local diagnostics |
| Space and row-wise time | `GTWR` | `SGTWR`, `MGTWR` |
| Snapshot/stage history | `STWR` | parameter sensitivity across stages |
| Global and local effects | global regression + `MixedGWR` | theory-supported variable partition |
| Local variable selection | `GWLasso` | stability and resampling analysis |
| Local multivariate structure | `GWSS`, `GWPCA` | local classification with `GWDA` |
| Geography plus functional similarity | `SGWR` | `SGTWR` or research `LGGWR` |
| Contiguous spatial mechanisms | standard GWR | research `GRGWR` |

Read the full [model selection guide](https://hujinghaoabcd.github.io/pyGWRx/getting-started/choosing-a-model/) and [19-model handbook](https://hujinghaoabcd.github.io/pyGWRx/models/).

## Kernels, bandwidths, and distances

Built-in kernels:

```python
from pygwrx.core import (
    gaussian_kernel,
    bisquare_kernel,
    exponential_kernel,
    tricube_kernel,
    boxcar_kernel,
)
```

- **Fixed bandwidth:** a distance in the chosen coordinate metric.
- **Adaptive bandwidth:** an integer neighbour count; the corresponding distance changes by focal location.
- **Compact kernels:** bisquare, tricube, and boxcar assign exact zero weight outside the local threshold.
- **Continuous kernels:** Gaussian and exponential retain positive weights with distance decay.

Report coordinate reference system, distance metric, kernel, fixed/adaptive mode, selection criterion, search range, and final bandwidth. For multiscale models, report every parameter-specific scale.

## Public functions and examples

The project contains **174 public API symbols**:

| Namespace | Scope |
|---|---|
| `pygwrx.models` | models and typed prediction-result objects |
| `pygwrx.core` | kernels, distances, bandwidths, optimization, solvers, metrics, base classes |
| `pygwrx.diagnostics` | model, residual, influence, inference, collinearity, temporal, weight, regime diagnostics |
| `pygwrx.plotting` | coefficient, residual, comparison, temporal, decomposition, latent-geometry, and regime plots |
| `pygwrx.io` | data conversion, persistence, and dataset registry |

Every symbol has:

- a generated API entry with signature and full docstring;
- a purpose summary and import path;
- a link to the maintained example that exercises it;
- the complete example source embedded in the relevant API page;
- a row in `examples/API_COVERAGE.json` and `.csv`.

Validate the contract:

```bash
python tools/generate_api_docs.py
python tools/generate_example_docs.py
python examples/validate_coverage.py
```

Run examples:

```bash
python -m pip install -e ".[all,test]"
python examples/run_all.py
```

Example inventory:

- 19 model examples
- 8 core numerical examples
- 5 diagnostics examples
- 6 plotting examples
- 4 I/O examples
- 3 end-to-end workflows

## Documentation

The MkDocs site includes:

- Getting Started and data contracts
- detailed English guides for all 19 models
- detailed Chinese guides for all 19 models
- core, diagnostics, plotting, and I/O function manuals
- full source for all 45 examples
- a 47-figure visualization gallery
- generated API pages for all 174 symbols
- complete algorithm encyclopedia and original-model monographs
- development, testing, release, citation, and API-stability guidance

Build locally:

```bash
python -m pip install -e ".[docs]"
python tools/generate_api_docs.py
python tools/generate_example_docs.py
mkdocs serve
# open http://127.0.0.1:8000

mkdocs build --strict --clean
```

## Validation and project status

Version **0.1.2** is an **Alpha research release**. The supported public tree is documented and represented by examples, but model capabilities are intentionally heterogeneous and 0.x APIs may evolve.

Current stable-suite baseline:

```text
363 passed
```

This local Linux/Python 3.13 result includes 360 non-reference tests, including a frozen self-contained MGTWR numerical fixture, and 3 independently maintained GWGLM comparisons from the optional `reference` extra. The 12-combination operating-system/Python matrix is configured as a blocking GitHub Actions workflow and must not be described as passed until the corresponding remote run succeeds.

## Data and legal notice

The development tree contains third-party example datasets. A literature citation, permission for academic analysis, and permission to redistribute the data inside a Python wheel are different legal questions. Verify the original licence and attribution requirements of every dataset before public redistribution.

The MIT licence covers pyGWRx-owned source code; it does not automatically relicense third-party datasets or dependencies.

## Project layout

```text
pyGWRx/
├── src/pygwrx/          # formal package
├── tests/               # stable test suite
├── examples/            # 45 runnable examples + coverage manifests
├── docs/                # English and Chinese manuals, API, theory, gallery
├── tools/               # API/documentation generators
├── pyproject.toml       # package metadata and optional extras
├── mkdocs.yml           # documentation layout
├── README.md            # English project overview
└── README.zh.md         # Chinese project overview
```

Historical prototypes, previous examples, previous documentation, legacy GTWR reporting code, and non-release audit materials are kept in separate archives rather than the formal package tree.

## Citation, author, and licence

- Author: **Jinghao Hu**
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Source licence: [MIT](LICENSE)
- Documentation: <https://hujinghaoabcd.github.io/pyGWRx/>


## Real-data five-minute start

```python
from pygwrx import GWR
from pygwrx.io import load_columbus

data = load_columbus(return_type="dict")
model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(
    data["data"], data["target"], data["coords"]
)
print(model.summary())
```

pyGWRx uses task-specific spatial modelling contracts; it does not promise
`scikit-learn` cloning, pipelines, `get_params()`, or `set_params()`. Bundled
datasets retain their upstream licenses; see `DATA_LICENSES.md`.
