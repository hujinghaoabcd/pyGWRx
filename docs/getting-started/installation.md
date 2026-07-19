# Installation

## Requirements

- CPython **3.11–3.14**
- NumPy, SciPy, pandas, Matplotlib, GeoPandas, and Shapely for the base package
- Python 3.12 or 3.13 is recommended for local development

## Source installation

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The base install includes Matplotlib, GeoPandas, and Shapely so plotting, mapping, GeoDataFrame workflows, and MGTWR are available immediately. scikit-learn and PyArrow remain optional.

## Optional dependency groups

| Extra | Install command | Provides |
|---|---|---|
| `ml` | `python -m pip install -e ".[ml]"` | GWLasso, GWPCA, and GRGWR dependencies |
| `parquet` | `python -m pip install -e ".[parquet]"` | Parquet and GeoParquet persistence |
| `all` | `python -m pip install -e ".[all]"` | all remaining user-facing optional features |
| `test` | `python -m pip install -e ".[test]"` | runtime test environment |
| `dev` | `python -m pip install -e ".[dev]"` | lint, test, typing, build, and release tools |
| `docs` | `python -m pip install -e ".[docs]"` | documentation toolchain |
| `reference` | `python -m pip install -e ".[reference]"` | external numerical-reference packages used by comparison tests |

Optional imports for scikit-learn and PyArrow are lazy. Plotting, geospatial support, and MGTWR are part of the standard installation.

## Isolated environment

=== "venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
    python -m pip install -e ".[all]"
    ```

=== "conda"

    ```bash
    conda create -n pygwrx python=3.13
    conda activate pygwrx
    python -m pip install -e ".[all]"
    ```

## Verify

```bash
python -c "import pygwrx; print(pygwrx.__version__)"
python tools/generate_api_docs.py
python examples/validate_coverage.py
```

## BLAS/OpenMP stability

For reproducible CI and local runs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Coordinate units

Do not treat longitude/latitude degrees as Euclidean metres. Project to an appropriate CRS or select a distance treatment consistent with the study region.
