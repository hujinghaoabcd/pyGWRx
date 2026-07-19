# Compare local regression models

A useful comparison changes one modelling mechanism at a time. The maintained workflow fits standard GWR, robust GWR, and locally compensated ridge GWR on the same data and neighbourhood size.

## Maintained workflow

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Compare standard, robust, and ridge-compensated local regressions."""

from pygwrx import GWR, LCRGWR, RGWR
from pygwrx.diagnostics import model_diagnostic_summary

from _common import spatial_regression

X, y, coords = spatial_regression()
models = {
    "GWR": GWR(bandwidth=24, adaptive=True),
    "RGWR": RGWR(bandwidth=24, adaptive=True, max_iter=5),
    "LCRGWR": LCRGWR(bandwidth=24, adaptive=True),
}
for name, model in models.items():
    model.fit(X, y, coords)
    summary = model_diagnostic_summary(model)
    print(name, summary)
```

## Interpretation

- **GWR** is the reference local surface.
- **RGWR** asks whether high-residual observations materially change the surface.
- **LCRGWR** asks whether local ill-conditioning destabilizes the surface.

Keep kernel, bandwidth, coordinate system, and data fixed when comparing these mechanisms. Then compare:

- global diagnostics and ENP;
- held-out prediction where supported;
- robust weights and outlier locations;
- local condition numbers and local lambdas;
- coefficient sign/magnitude stability;
- residual spatial pattern;
- scientific interpretability.

A specialized model is justified only when it addresses a diagnosed problem and improves out-of-sample or inferential behaviour.
