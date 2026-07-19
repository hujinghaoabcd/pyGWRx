# Kernels, distances, and bandwidths

The neighbourhood is the central modelling decision in geographically weighted methods. A kernel says how influence decays; a bandwidth says the scale of that decay; a distance metric says what “near” means.

## Built-in kernels

For normalized distance $r=d/h$:

| Kernel | Weight | Support | Interpretation |
|---|---|---|---|
| Gaussian | $\exp(-r^2/2)$ | infinite | smooth long tail |
| Exponential | $\exp(-r)$ | infinite | sharper long tail |
| Bisquare | $(1-r^2)^2$ for $r<1$ | compact | smooth local window |
| Tricube | $(1-|r|^3)^3$ for $|r|<1$ | compact | smooth compact window |
| Boxcar | $1$ for $r<1$ | compact | equal-weight local window |

```python
import numpy as np
from pygwrx.core import (
    gaussian_kernel,
    bisquare_kernel,
    exponential_kernel,
    tricube_kernel,
    boxcar_kernel,
    get_kernel_function,
)

d = np.array([0.0, 0.5, 1.0, 2.0])
print(gaussian_kernel(d, bandwidth=1.5))
print(bisquare_kernel(d, bandwidth=1.5))
print(get_kernel_function("tricube"))
```

Compact support can improve interpretability and computation because distant observations receive exact zero weight. Infinite-support kernels still use all observations, although distant weights may be tiny.

## Fixed bandwidth

```python
from pygwrx import GWR
model = GWR(kernel="gaussian", bandwidth=2000.0, adaptive=False)
```

The value uses the distance units of the coordinate metric. A 2,000-unit bandwidth means 2,000 metres only when the coordinates are projected in metres.

Use a fixed bandwidth when a constant physical interaction range is meaningful and sampling density is reasonably uniform.

## Adaptive bandwidth

```python
model = GWR(kernel="bisquare", bandwidth=40, adaptive=True)
```

The value is an integer neighbour count. The local physical radius is the distance to the selected neighbour and therefore changes by location.

Use an adaptive bandwidth when sampling density varies substantially or a minimum local sample size is required.

## Distance metrics

```python
from pygwrx.core import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
    haversine_distance,
    compute_distance_matrix,
)
```

Distance choice changes the neighbourhood geometry. Project coordinates for Euclidean modelling. Haversine distance is appropriate only when the model and units are consistent with spherical coordinates.

## Automatic selection

```python
model = GWR(
    kernel="bisquare",
    bandwidth="aicc",
    bandwidth_range=(20, 60),
    adaptive=True,
)
model.fit(X, y, coords)
print(model.bandwidth_)
```

The core layer also exposes selectors:

```python
from pygwrx.core import (
    CrossValidationSelector,
    AICSelector,
    BICSelector,
    get_bandwidth_selector,
)
```

Selection criteria answer different questions:

- CV focuses on prediction error under the implemented leave-one-out logic.
- AIC/AICc balance fit and effective complexity.
- BIC penalizes complexity more strongly.

Not every criterion is implemented for every model family.

## Search bounds and boundary solutions

A selected value at the lower or upper search limit should trigger investigation:

- expand the range;
- inspect the objective curve;
- verify fixed/adaptive units;
- check minimum local rank;
- determine whether the process is effectively global or extremely local.

## Multiscale and spatiotemporal scales

- MGWR: one spatial bandwidth per design column.
- GTWR: a space-time balance plus a kernel bandwidth.
- SGTWR: separate spatial bandwidth, temporal bandwidth, and alpha.
- MGTWR: one spatial bandwidth and temporal scale per coefficient.
- STWR: stage history, alpha, theta, and evolving historical bandwidths.

These parameters interact. Report them together rather than interpreting one in isolation.

## Reporting checklist

- coordinate reference system and distance units;
- distance metric;
- kernel;
- fixed or adaptive mode;
- selection criterion;
- search range/candidates;
- optimizer, tolerance, and iterations;
- final bandwidth or scale vector;
- sensitivity to plausible alternatives.

See the [Core numerical guide](core-numerics.md), [bandwidth API](../api/core/bandwidth.md), and [kernel API](../api/core/kernels.md).
