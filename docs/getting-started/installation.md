# Installation

## Requirements

- CPython **3.11–3.14**
- A recent `pip`
- NumPy, SciPy, pandas, Matplotlib, GeoPandas, and Shapely are installed with the base package

Python 3.12 or 3.13 is recommended for local development.

## Install from PyPI

Install or upgrade the latest published release:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade pyGWRx
```

For a reproducible environment, pin the current release explicitly:

```bash
python -m pip install "pyGWRx==0.1.2"
```

The base installation includes Matplotlib, GeoPandas, and Shapely, so plotting, mapping, GeoDataFrame workflows, and MGTWR are available immediately.

## Optional dependency groups

Install only the additional features required by your workflow:

| Extra | Install command | Provides |
|---|---|---|
| `ml` | `python -m pip install "pyGWRx[ml]"` | scikit-learn support used by GWLasso, GWPCA, and GRGWR |
| `parquet` | `python -m pip install "pyGWRx[parquet]"` | PyArrow persistence for Parquet and GeoParquet |
| `all` | `python -m pip install "pyGWRx[all]"` | all user-facing optional features |

The `test`, `dev`, `docs`, and `reference` extras are intended for contributors working from a source checkout rather than ordinary package users.

## Use an isolated environment

=== "Windows PowerShell"

    ```powershell
    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install --upgrade pyGWRx
    ```

=== "Linux and macOS"

    ```bash
    python3.13 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install --upgrade pyGWRx
    ```

=== "conda"

    ```bash
    conda create -n pygwrx python=3.13
    conda activate pygwrx
    python -m pip install --upgrade pyGWRx
    ```

## Verify the installed package

These checks run entirely against the installed distribution and do not require a repository checkout:

```bash
python -c "import pygwrx; print(pygwrx.__version__)"
python -c "from pygwrx import GWR, MGWR, MGTWR; print('Core model imports passed')"
python -c "from pygwrx.io import load_dataset; X, y, coords = load_dataset('columbus', return_type='arrays'); print(X.shape, y.shape, coords.shape)"
python -m pip check
```

For version 0.1.2, the dataset command should print:

```text
(49, 2) (49,) (49, 2)
```

## Install from source for development

Clone the repository only when you plan to modify pyGWRx, run its complete test suite, or build the documentation:

```bash
git clone https://github.com/hujinghaoabcd/pyGWRx.git
cd pyGWRx
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Additional contributor environments can be installed as needed:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[reference]"
python -m pip install -e ".[all,test,dev,docs,reference]"
```

## BLAS/OpenMP stability

For reproducible CI and local runs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

On Windows PowerShell, set the same variables with `$env:NAME = "1"`.

## Coordinate units

Do not treat longitude/latitude degrees as Euclidean metres. Project to an appropriate CRS or select a distance treatment consistent with the study region.
