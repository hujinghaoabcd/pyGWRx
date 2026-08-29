# GWR External Reference Validation

Generated from the deterministic 40-point calibration fixture and 5 independent prediction locations.

## Strict like-for-like comparisons

| Reference | Checks | Worst max absolute difference | Metric/case |
|---|---:|---:|---|
| mgwr | 91 | 6.029953e-06 | adaptive_bisquare_v2 / aic |
| GWmodel | 42 | 9.683034e-07 | adaptive_bisquare_v2 / t_values |
| spgwr | 10 | 4.018937e-08 | fixed_bisquare_v2 / params |

## Known semantic differences

- GWmodel Local_R2 is not numerically identical to the mgwr/spgwr/PyGWRx local weighted R² convention and is reported separately.
- GWmodel AIC and BIC labels use formulas that differ from the RSS/trace(S) formulas used by PyGWRx/mgwr; AICc is directly comparable and is tested strictly.
- spgwr adaptive bandwidth is supplied as a sample proportion and resolves local radii differently from integer-k adaptive bandwidths; adaptive results are therefore semantic cross-checks, not strict equality tests.
- With sigma2_v1=True, mgwr uses a different effective-parameter convention for adjusted R²; that diagnostic is archived as a definition difference rather than a strict error.
- Bandwidth-selection validation is handled separately by the controlled k=4..40 criterion-curve report, which removes optimizer/default-range effects.
