# GWR external-reference validation

This directory contains reproducible generators for validating `pygwrx.GWR` against independently maintained GWR implementations.

## References

- `mgwr` 2.2.1 (Python)
- `GWmodel` (R; version recorded in generated JSON)
- `spgwr` (R; version recorded in generated JSON)

The deterministic calibration data live in `tests/reference_data/gwr/input.csv`; independent prediction targets live in `tests/reference_data/gwr/prediction.csv`.

## Validation policy

Only quantities with matching mathematical definitions are used as strict numerical references. A feature is not forced into a three-package comparison when package semantics differ.

Strict/shared comparisons include, where available:

- fixed Gaussian and bisquare calibration;
- adaptive Gaussian and bisquare calibration;
- local intercepts and slopes;
- fitted values and residuals;
- local R-squared;
- smoother and effective-complexity diagnostics when definitions align;
- coefficient standard errors and t statistics under aligned variance conventions;
- target-location local recalibration and prediction;
- bandwidth selection as a separate semantic/reference check.

`spgwr` adaptive bandwidths are proportions rather than integer neighbour counts. Its adaptive results are retained as a secondary semantic reference rather than treated as bit-for-bit equivalents of integer-k implementations.

For Gaussian weights in `spgwr`, the generator deliberately uses `gwr.Gauss`; the older `gwr.gauss` function uses a different exponent convention and is not mathematically equivalent to the pyGWRx Gaussian kernel.

## Regeneration

Python reference:

```bash
python -m pip install mgwr==2.2.1 pandas numpy
python tools/reference/gwr/generate_mgwr.py
```

R references:

```r
install.packages(c("jsonlite", "sp", "spgwr", "GWmodel"))
```

```bash
Rscript tools/reference/gwr/generate_r_references.R
```

Generated JSON files are frozen under `tests/reference_data/gwr/` and consumed by tests marked `reference`. Normal CI therefore validates against frozen outputs without requiring R at test time.
