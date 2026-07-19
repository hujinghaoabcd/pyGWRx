# pyGWRx runnable examples

The example tree is built around the current public API. It contains **45 isolated scripts** and maps all **174 public symbols** to at least one runnable example.

| Directory | Scripts | Coverage |
|---|---:|---|
| `models/` | 19 | one script for every supported public model |
| `core/` | 8 | base classes, kernels, distances, validation, solvers, metrics, optimization, bandwidth selection |
| `diagnostics/` | 5 | model, residual, inference, collinearity, temporal, weight, and regime diagnostics |
| `plotting/` | 6 | every public plotting function |
| `io/` | 4 | bundled datasets, tabular/geographic conversion, and persistence |
| `workflows/` | 3 | end-to-end GWR, model comparison, and spatiotemporal workflows |

## Start with a bundled real dataset

```bash
python examples/models/01_gwr.py
python examples/workflows/01_end_to_end_gwr.py
```

Every maintained script resolves the project source and `examples/_common.py`
from its own file location, so it can be launched from the project root, an IDE,
or another current working directory.

## Install the example environment

```bash
python -m pip install -e ".[all,test]"
```

## Run one script

Run examples from the project root so `_common.py` and source paths resolve consistently:

```bash
python examples/models/01_gwr.py
python examples/diagnostics/02_inference_and_collinearity.py
python examples/plotting/06_lggwr_and_grgwr.py
```

## Run all scripts

```bash
python examples/run_all.py
```

`run_all.py` starts each script in a separate process, configures Matplotlib for non-interactive rendering, and limits BLAS/OpenMP threads to one. Generated outputs are written under `examples/output/` and can be deleted safely.

## Verify public API coverage

```bash
python tools/generate_api_docs.py
python tools/generate_example_docs.py
python examples/validate_coverage.py
```

The generator writes:

- `examples/API_COVERAGE.json`
- `examples/API_COVERAGE.csv`
- grouped API pages under `docs/api/`

Validation fails when a public symbol is missing an example, a stale symbol remains in the manifest, or an example no longer imports its assigned API.

## Capability notes

- MGWR and MGTWR examples demonstrate calibration results and intentionally catch the unsupported independent-target prediction error.
- GWLasso, GWPCA, and GRGWR require the `ml` extra.
- plotting examples require `plot`; geographic examples require `geo`.
- optional `mgwr` and `spglm` packages are only needed for GWGLM numerical-reference tests, not for the examples or runtime GWGLM implementation.

The rendered catalog is available in the [documentation](https://hujinghaoabcd.github.io/pyGWRx/examples/).
