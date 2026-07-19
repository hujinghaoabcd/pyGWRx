# Troubleshooting

## Import or optional-dependency error

Install the extra named in the message:

```bash
python -m pip install -e ".[ml]"
```

Matplotlib, GeoPandas, Shapely, and MGTWR are included in the standard installation. Only scikit-learn, PyArrow, and reference-comparison packages require extras.

## Shape or alignment error

Check that rows correspond across all inputs:

```python
print(X.shape, len(y), coords.shape)
```

For row-wise time models, `len(times)` must match. For STWR, validate every stage separately.

## NaN or infinite values

The validation layer rejects non-finite inputs. Handle missingness before fitting and document the rule. Do not replace missing values with zero unless zero has the intended scientific meaning.

## Coordinates produce strange bandwidths

- confirm the CRS;
- do not interpret longitude/latitude degrees as metres;
- check fixed versus adaptive mode;
- inspect duplicated coordinates;
- verify the selected distance metric.

## Local solve fails or coefficients explode

Possible causes:

- bandwidth too small for the number of design columns;
- local rank deficiency;
- duplicate or constant predictors;
- severe local collinearity;
- extreme scaling differences;
- a near-empty class/event neighbourhood.

Increase the neighbourhood, remove redundant variables, standardize where appropriate, or use LCRGWR when local collinearity is itself part of the analysis.

## Bandwidth optimum lies on a boundary

Expand the range and inspect the objective values. Confirm that the criterion is supported and that fixed/adaptive units are correct. A boundary result can also indicate a nearly global or extremely local process.

## Model does not converge

- increase `max_iter` only after checking the objective history;
- standardize features;
- use a more stable initialization;
- loosen an unrealistically strict tolerance;
- reduce candidate-grid complexity;
- inspect local event/class counts;
- run multiple restarts for LGGWR/GRGWR.

## Prediction raises `NotImplementedError`

This is expected for independent-target prediction in MGWR and MGTWR. Use calibration results only. Do not reinterpret fitted values as held-out predictions.

## An argument from an earlier development snapshot is rejected

Inactive or unimplemented constructor arguments are not retained as public API. Check the current generated signature and migration notes rather than relying on a pre-release example.

## Map and table are misaligned

Join with a stable ID and validate one-to-one cardinality. Row order can change during GIS operations.

## Plotting creates no window

pyGWRx plotting functions return Matplotlib objects and do not call `plt.show()`:

```python
fig, ax = plot_coefficient_map(model, feature="x1")
fig.savefig("map.png")
# or plt.show()
```

## Documentation build fails

```bash
python -m pip install -e ".[docs]"
python tools/generate_api_docs.py
mkdocs build --strict --clean
```

Generated API files should not be manually edited.

## Full test process hangs or does not exit

Limit numerical-library threads:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

On PowerShell:

```powershell
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
```

Then rerun in a clean environment.
