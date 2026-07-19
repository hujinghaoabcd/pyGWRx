# Original research models: LGGWR and GRGWR

LGGWR and GRGWR are original research implementations. Their documentation includes stronger reporting and sensitivity requirements than a conventional fixed algorithm because their learned geometry or regime structure can depend on initialization and tuning.

## LGGWR workflow

1. Fit standard GWR as a geographic baseline.
2. Define contextual attributes that are scientifically available and not outcome proxies.
3. Standardize geometry inputs.
4. Run multiple restarts.
5. Inspect objective history, latent coordinates, metric matrix, and neighbourhood changes.
6. Compare held-out prediction and coefficient stability with GWR.
7. Report latent dimension, constraints, regularization, learning rate, initialization, and restart variability.

```python
from pygwrx import LGGWR

model = LGGWR(
    latent_dim=2,
    bandwidth=2.5,
    select_bandwidth=False,
    n_restarts=3,
    random_state=0,
).fit(X, y, coords, attributes)
```

See the [LGGWR handbook](../models/lg-gwr.md) and [LG-GWR monograph](../theory/lg-gwr-monograph.zh.md).

## GRGWR workflow

1. Fit GWR and inspect whether coefficient changes appear abrupt rather than smooth.
2. Specify a defensible regime-count range.
3. Build/verify the spatial-neighbour graph.
4. Run multiple initializations.
5. Check connectivity, minimum regime size, boundary stability, and convergence.
6. Compare regime-conditioned prediction with GWR and simpler regional models.
7. Report graph construction, boundary penalty, regime count, random seeds, and sensitivity.

```python
from pygwrx import GRGWR

model = GRGWR(
    n_regimes=3,
    bandwidth=24,
    enforce_connectivity=True,
    random_state=0,
).fit(X, y, coords)
```

See the [GRGWR handbook](../models/gr-gwr.md) and [GR-GWR monograph](../theory/gr-gwr-monograph.zh.md).
