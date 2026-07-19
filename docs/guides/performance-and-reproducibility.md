# Performance and reproducibility

Local models can fit a regression at every observation and repeat this process during bandwidth selection.

## Practical controls

- validate semantics with explicit bandwidths first
- restrict automatic search bounds
- avoid storing full weights unless required
- compare ScalableGWR with standard GWR on a manageable subset
- set random seeds for bootstrap, regime, and latent-geometry procedures

## Numerical threads

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Reproducibility record

Report package and dependency versions, model parameters, kernel, bandwidth settings, CRS and time units, feature scaling, missing-data handling, random seed, convergence, and evaluation split.
