# Model examples

One isolated, runnable script for every supported public model.

This page embeds **19** maintained scripts. The code shown here is read directly from `examples/models/`, so the documentation and executable source cannot silently diverge.

!!! tip "How to use this catalog"
    Read the purpose and APIs first, run the exact command, then use the inspection note to decide which output matters. For conceptual interpretation, follow the linked model or function guide rather than reading code alone.

## `01_gwr.py`

**Purpose.** Load a bundled real dataset, fit GWR, inspect it, and predict.

**Public APIs exercised.** `GWR`, `GWRPredictionResult`, `load_columbus`

**Environment.** base installation.

**Run.** `python examples/models/01_gwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/01_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Load a bundled real dataset, fit GWR, inspect it, and predict."""

from __future__ import annotations

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pygwrx import GWR, GWRPredictionResult
from pygwrx.io import load_columbus

bundle = load_columbus(return_type="dict")
X = bundle["data"]
y = bundle["target"]
coords = bundle["coords"]

print("dataset=", bundle["description"])
print("features=", bundle["feature_names"])
print("license=", bundle["license"])

model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(X, y, coords)
print(model.summary())
print("score=", model.score(X, y, coords))

result = model.predict_result(X[:3], coords[:3])
assert isinstance(result, GWRPredictionResult)
print(result.to_frame())
```

## `02_mgwr.py`

**Purpose.** Fit MGWR with fixed variable-specific bandwidths.

**Public APIs exercised.** `MGWR`

**Environment.** base installation.

**Run.** `python examples/models/02_mgwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/mgwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/mgwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/02_mgwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit MGWR with fixed variable-specific bandwidths."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, spatial_regression

from pygwrx import MGWR

X, y, coords = spatial_regression(n=48, p=2)
model = MGWR(bandwidths=[24, 26, 28], adaptive=True, max_iter=8, tol=0.5).fit(
    X, y, coords, compute_inference=True
)
print_model_result(model)
try:
    model.predict(X.iloc[:2], coords.iloc[:2])
except NotImplementedError as exc:
    print("Expected MGWR prediction limitation:", exc)
```

## `03_rgwr.py`

**Purpose.** Fit robust GWR in automatic down-weighting mode.

**Public APIs exercised.** `RGWR`

**Environment.** base installation.

**Run.** `python examples/models/03_rgwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/rgwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/rgwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/03_rgwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit robust GWR in automatic down-weighting mode."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
from _common import print_model_result, spatial_regression

from pygwrx import RGWR

X, y, coords = spatial_regression()
y = y.copy()
y[[2, 20]] += np.array([5.0, -4.0])
model = RGWR(bandwidth=24, adaptive=True, max_iter=8).fit(X, y, coords)
print_model_result(model)
print("robust_weights=", model.robust_weights_[:8])
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

## `04_stwr.py`

**Purpose.** Fit STWR from multiple observation snapshots.

**Public APIs exercised.** `STWR`, `STWRPredictionResult`

**Environment.** base installation.

**Run.** `python examples/models/04_stwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/stwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/stwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/04_stwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit STWR from multiple observation snapshots."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, stwr_stages

from pygwrx import STWR, STWRPredictionResult

X_list, y_list, coords_list, intervals = stwr_stages()
model = STWR(
    spatial_bandwidth=10,
    adaptive=True,
    alpha=0.3,
    theta=0.0,
    tick_nums=2,
    store_weights=True,
).fit(X_list, y_list, coords_list, intervals)
print_model_result(model)
result = model.predict_result(
    X_list[-1].iloc[:3],
    coords_list[-1].iloc[:3],
    reference_y=y_list[-1][:3],
)
assert isinstance(result, STWRPredictionResult)
print(result.to_frame())
```

## `05_gtwr.py`

**Purpose.** Fit and predict with geographically and temporally weighted regression.

**Public APIs exercised.** `GTWR`, `GTWRPredictionResult`

**Environment.** base installation.

**Run.** `python examples/models/05_gtwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gtwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gtwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/05_gtwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit and predict with geographically and temporally weighted regression."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, temporal_regression

from pygwrx import GTWR, GTWRPredictionResult

X, y, coords, times = temporal_regression()
model = GTWR(kernel="bisquare", bandwidth=24, adaptive=True, lambda_st=0.3).fit(
    X, y, coords, times
)
print_model_result(model)
print("score=", model.score(X, y, coords, times=times))
result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
assert isinstance(result, GTWRPredictionResult)
print(result.to_frame())
```

## `06_gwglm.py`

**Purpose.** Fit Gaussian, binomial, and Poisson GWGLM families.

**Public APIs exercised.** `GWGLM`, `GWGLMPredictionResult`

**Environment.** base installation.

**Run.** `python examples/models/06_gwglm.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gwglm.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gwglm.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/06_gwglm.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit Gaussian, binomial, and Poisson GWGLM families."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
from _common import count_regression, print_model_result, spatial_regression

from pygwrx import GWGLM, GWGLMPredictionResult

X, y, coords = spatial_regression(p=2)
gaussian = GWGLM(family="gaussian", bandwidth=24, adaptive=True).fit(X, y, coords)
print_model_result(gaussian)

binary = (y > np.median(y)).astype(int)
binomial = GWGLM(family="binomial", bandwidth=24, adaptive=True).fit(X, binary, coords)
binomial_result = binomial.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(binomial_result, GWGLMPredictionResult)
print(binomial_result.to_frame())

Xc, counts, coordsc, exposure = count_regression()
poisson = GWGLM(family="poisson", bandwidth=24, adaptive=True).fit(
    Xc, counts, coordsc, exposure=exposure
)
print(
    "poisson means=",
    poisson.predict(Xc.iloc[:3], coordsc.iloc[:3], exposure=exposure[:3]),
)
```

## `07_gw_lasso.py`

**Purpose.** Fit geographically weighted Lasso with a fixed local penalty.

**Public APIs exercised.** `GWLasso`

**Environment.** `pip install -e ".[ml]"`.

**Run.** `python examples/models/07_gw_lasso.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gw-lasso.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gw-lasso.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/07_gw_lasso.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geographically weighted Lasso with a fixed local penalty."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, spatial_regression

from pygwrx import GWLasso

X, y, coords = spatial_regression(n=48, p=3)
model = GWLasso(
    bandwidth=24, adaptive=True, alpha=0.06, max_iter=1000, random_state=0
).fit(X, y, coords)
print_model_result(model)
print("selection_frequency=", model.selection_frequency_)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

## `08_mixed_gwr.py`

**Purpose.** Fit a semiparametric Mixed GWR with global and local predictors.

**Public APIs exercised.** `MixedGWR`

**Environment.** base installation.

**Run.** `python examples/models/08_mixed_gwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/mixed-gwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/mixed-gwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/08_mixed_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit a semiparametric Mixed GWR with global and local predictors."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import mixed_regression, print_model_result

from pygwrx import MixedGWR

X, y, coords = mixed_regression()
model = MixedGWR(
    bandwidth=28,
    adaptive=True,
    global_vars=["global_x"],
    local_vars=["local_x"],
    intercept_fixed=True,
).fit(X, y, coords, compute_enp=False)
print_model_result(model)
print("global_coefficients=", model.coef_global_)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

## `09_gwpca.py`

**Purpose.** Fit GWPCA, inspect local loadings, and transform observations.

**Public APIs exercised.** `GWPCA`

**Environment.** `pip install -e ".[ml]"`.

**Run.** `python examples/models/09_gwpca.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gwpca.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gwpca.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/09_gwpca.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit GWPCA, inspect local loadings, and transform observations."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, spatial_regression

from pygwrx import GWPCA

X, _, coords = spatial_regression(n=48, p=3)
model = GWPCA(n_components=2, bandwidth=24, adaptive=True).fit(
    X, coords, compute_cv=True
)
print_model_result(model)
print("scores_shape=", model.transform(X, coords).shape)
print("explained_variance_first_location=", model.local_pv_[0])
```

## `10_gwda.py`

**Purpose.** Fit geographically weighted discriminant analysis.

**Public APIs exercised.** `GWDA`

**Environment.** base installation.

**Run.** `python examples/models/10_gwda.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gwda.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gwda.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/10_gwda.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geographically weighted discriminant analysis."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import classification_data

from pygwrx import GWDA

X, y, coords = classification_data()
model = GWDA(bandwidth=28, adaptive=True, quadratic=False).fit(X, y, coords)
print(model.summary())
print("classes=", model.classes_)
print("predictions=", model.predict(X.iloc[:5], coords.iloc[:5]))
print("probabilities=", model.predict_proba(X.iloc[:5], coords.iloc[:5]))
```

## `11_gwss.py`

**Purpose.** Compute geographically weighted summary statistics.

**Public APIs exercised.** `GWSS`

**Environment.** base installation.

**Run.** `python examples/models/11_gwss.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gwss.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gwss.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/11_gwss.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Compute geographically weighted summary statistics."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import spatial_regression

from pygwrx import GWSS

X, _, coords = spatial_regression(n=48, p=3)
model = GWSS(bandwidth=24, adaptive=True, quantile=True).fit(X, coords)
print(model.summary())
print("local_means_shape=", model.local_mean_.shape)
print("local_correlation_pairs=", sorted(model.local_corr_))
print("first_correlation_shape=", next(iter(model.local_corr_.values())).shape)
```

## `12_scalable_gwr.py`

**Purpose.** Fit scalable GWR with a fixed multiscale-kernel approximation.

**Public APIs exercised.** `ScalableGWR`

**Environment.** base installation.

**Run.** `python examples/models/12_scalable_gwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/scalable-gwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/scalable-gwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/12_scalable_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit scalable GWR with a fixed multiscale-kernel approximation."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, spatial_regression

from pygwrx import ScalableGWR

X, y, coords = spatial_regression(n=54, p=2)
model = ScalableGWR(
    bandwidth=24, optimize_bandwidth=False, polynomial=4, random_state=0
).fit(X, y, coords)
print_model_result(model)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

## `13_lcr_gwr.py`

**Purpose.** Fit locally compensated ridge GWR for collinear predictors.

**Public APIs exercised.** `LCRGWR`

**Environment.** base installation.

**Run.** `python examples/models/13_lcr_gwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/lcr-gwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/lcr-gwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/13_lcr_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit locally compensated ridge GWR for collinear predictors."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import collinear_regression, print_model_result

from pygwrx import LCRGWR

X, y, coords = collinear_regression()
model = LCRGWR(bandwidth=28, adaptive=True, cn_thresh=15.0, lambda_adjust=True).fit(
    X, y, coords
)
print_model_result(model)
print("local_condition_numbers=", model.local_condition_numbers_[:5])
print("local_lambdas=", model.local_lambdas_[:5])
```

## `14_bootstrap_gwr.py`

**Purpose.** Run coefficient-wise bootstrap tests for spatial variability.

**Public APIs exercised.** `BootstrapGWR`

**Environment.** base installation.

**Run.** `python examples/models/14_bootstrap_gwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/bootstrap-gwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/bootstrap-gwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/14_bootstrap_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run coefficient-wise bootstrap tests for spatial variability."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, spatial_regression

from pygwrx import BootstrapGWR

X, y, coords = spatial_regression(n=42, p=2)
model = BootstrapGWR(
    bandwidth=22,
    adaptive=True,
    n_bootstrap=9,
    reselect_bandwidth=False,
    store_local_bootstrap=True,
    random_state=0,
).fit(X, y, coords)
print_model_result(model)
print("modified_pvalues=", model.modified_p_values_)
print("localized_p_values_shape=", model.localized_p_values_.shape)
```

## `15_sgwr.py`

**Purpose.** Fit similarity and geographically weighted regression.

**Public APIs exercised.** `SGWR`

**Environment.** base installation.

**Run.** `python examples/models/15_sgwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/sgwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/sgwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/15_sgwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit similarity and geographically weighted regression."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, spatial_regression

from pygwrx import SGWR

X, y, coords = spatial_regression(n=48, p=3)
model = SGWR(
    bandwidth=24,
    adaptive=True,
    alpha=0.45,
    similarity_vars=["x1", "x2"],
    store_weights=True,
).fit(X, y, coords)
print_model_result(model)
print("combined_weights_shape=", model.combined_weights_.shape)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

## `16_sgtwr.py`

**Purpose.** Fit similarity and geographically-temporally weighted regression.

**Public APIs exercised.** `SGTWR`, `SGTWRPredictionResult`

**Environment.** base installation.

**Run.** `python examples/models/16_sgtwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/sgtwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/sgtwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/16_sgtwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit similarity and geographically-temporally weighted regression."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, temporal_regression

from pygwrx import SGTWR, SGTWRPredictionResult

X, y, coords, times = temporal_regression(n=48, p=3)
model = SGTWR(
    spatial_bandwidth=24,
    temporal_bandwidth=2.0,
    adaptive=True,
    alpha=0.5,
    similarity_vars=["x1", "x2"],
    store_weights=True,
).fit(X, y, coords, times)
print_model_result(model)
print("combined_weights_shape=", model.combined_weights_.shape)
result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
assert isinstance(result, SGTWRPredictionResult)
print(result.to_frame())
```

## `17_mgtwr.py`

**Purpose.** Fit the self-contained multiscale geographically and temporally weighted regression.

**Public APIs exercised.** `MGTWR`

**Environment.** base installation.

**Run.** `python examples/models/17_mgtwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/mgtwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/mgtwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/17_mgtwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit the self-contained multiscale geographically and temporally weighted regression."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, temporal_regression

from pygwrx import MGTWR

X, y, coords, times = temporal_regression(n=20, p=2)
model = MGTWR(
    bandwidths=[12, 12, 12],
    taus=[1.0, 1.0, 1.0],
    adaptive=True,
    calculate_inference=False,
).fit(X, y, coords, times)
print_model_result(model)
print("spatial_bandwidths=", model.bandwidths_)
print("temporal_scales=", model.taus_)
try:
    model.predict(X.iloc[:2], coords.iloc[:2], times[:2])
except NotImplementedError as exc:
    print("Expected MGTWR prediction limitation:", exc)
```

## `18_lg_gwr.py`

**Purpose.** Fit latent-geometry GWR with auxiliary contextual attributes.

**Public APIs exercised.** `LGGWR`, `LGGWRPredictionResult`

**Environment.** base installation.

**Run.** `python examples/models/18_lg_gwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/lg-gwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/lg-gwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/18_lg_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit latent-geometry GWR with auxiliary contextual attributes."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import latent_regression, print_model_result

from pygwrx import LGGWR, LGGWRPredictionResult

X, y, coords, attributes = latent_regression()
model = LGGWR(
    latent_dim=2, bandwidth=2.5, select_bandwidth=False, max_iter=8, random_state=0
).fit(X, y, coords, attributes)
print_model_result(model)
print("latent_coordinates_shape=", model.latent_coords_.shape)
result = model.predict_result(X.iloc[:3], coords.iloc[:3], attributes.iloc[:3])
assert isinstance(result, LGGWRPredictionResult)
print(result.to_frame())
```

## `19_gr_gwr.py`

**Purpose.** Fit geo-regime GWR and inspect connected spatial regimes.

**Public APIs exercised.** `GRGWR`, `GRGWRPredictionResult`

**Environment.** `pip install -e ".[ml]"`.

**Run.** `python examples/models/19_gr_gwr.py`

**What to inspect.** Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.

[Detailed model guide](../models/gr-gwr.md){ .md-button .md-button--primary }
[Chinese guide](../zh/models/gr-gwr.md){ .md-button }
[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/19_gr_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geo-regime GWR and inspect connected spatial regimes."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, regime_regression

from pygwrx import GRGWR, GRGWRPredictionResult

X, y, coords, truth = regime_regression(n=56)
model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
print_model_result(model)
print("regime_sizes=", model.regime_sizes_)
print(
    "truth_agreement_or_label_swap=",
    max((model.regimes_ == truth).mean(), (model.regimes_ != truth).mean()),
)
result = model.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(result, GRGWRPredictionResult)
print(result.to_frame())
```
