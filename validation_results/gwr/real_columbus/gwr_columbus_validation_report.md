# Columbus Real-Data GWR External Validation

Dataset: 49 Columbus, Ohio neighbourhoods; model `CRIME ~ INC + HOVAL`; coordinates `X`, `Y`.

Four calibration configurations are checked: fixed/adaptive × Gaussian/bisquare. Five geographically dispersed neighbourhoods (zero-based rows 0, 10, 20, 30, 40) are withheld and predicted from a 44-neighbourhood training fit.

## Strict numerical comparisons

| Reference | Version | Strict checks | Worst max absolute difference | Worst case/metric |
|---|---|---:|---:|---|
| mgwr | 2.2.1 | 91 | 1.529442e-05 | adaptive_bisquare_v2 / params |
| GWmodel | 2.4.1 | 42 | 2.109976e-06 | fixed_bisquare_v2 / params |
| spgwr | 0.6.37 | 10 | 2.109975e-06 | fixed_bisquare_v2 / params |

## Controlled adaptive-bisquare bandwidth validation

All integer candidates `k=4..49` are archived. `k=4` is retained as a transparent near-saturated boundary (`trace(S) ≈ n`); pyGWRx correctly reports non-finite AICc there. The table therefore reports both raw argmins and the `k>=5` diagnostic summary instead of silently deleting the boundary point.

| Criterion | Reference | Raw argmin py/ref | k>=5 argmin py/ref | k>=5 max abs diff | k>=5 max rel diff | Interpretation |
|---|---|---:|---:|---:|---:|---|
| cv_sse | mgwr | 11/11 | 11/11 | 1.538905e+02 | 1.028608e-03 | strict |
| cv_sse | GWmodel | 11/11 | 11/11 | 1.538911e+02 | 1.028611e-03 | strict |
| aic | mgwr | 4/4 | 5/5 | 3.614848e-05 | 1.533442e-07 | strict |
| aicc | mgwr | 24/24 | 24/24 | 1.010642e-02 | 4.231984e-06 | strict |
| aicc | GWmodel | 24/24 | 24/24 | 1.280206e-02 | 5.360758e-06 | strict |
| bic | mgwr | 4/4 | 5/5 | 1.966922e-05 | 6.094402e-08 | strict |
| bic | GWmodel | 4/4 | 5/5 | 5.289188e+01 | 1.960043e-01 | different_definition |

## Semantic boundaries

- Fixed Gaussian/bisquare comparisons against all three packages are strict like-for-like checks.
- On real data, pyGWRx adaptive neighbourhoods are numerically closest to GWmodel; mgwr remains very close, while spgwr uses a sample-proportion `q` and is therefore archived only as an adaptive semantic cross-check.
- GWmodel `Local_R2` and AIC/BIC conventions are not forced to equal pyGWRx where definitions differ.
- mgwr `sigma2_v1=True` adjusted-R² uses a different ENP convention; the distinction is explicitly preserved.
- Raw full external outputs are reproducible with the generator scripts; compact frozen fixtures omit large hat matrices to keep the repository lean.
