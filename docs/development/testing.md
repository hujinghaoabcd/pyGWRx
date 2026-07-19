# Testing

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Test layers include numerical utilities, model behaviour, edge cases, diagnostics, plotting, I/O round trips, static numerical fixtures, optional implementation comparisons, examples, and build/install smoke tests.

## Self-contained MGTWR numerical regression

The fixed-scale MGTWR regression fixture is stored at:

```text
tests/reference_data/mgtwr_fixed_gaussian_reference.json
```

It was produced once by an independent implementation outside the repository. The independent reference suite compares local coefficients, fitted values, residuals, coefficient-specific effective parameter counts, standard errors, t statistics, Gaussian information criteria, and iteration count. No external MGTWR package is imported, installed, or declared by this test.

```bash
python -m pytest tests/test_mgtwr.py -q
python tools/run_tests.py --batch 1
python tools/run_tests.py --batch 2
python tools/run_tests.py --batch 3
python -m pytest -q -m reference
```

## Optional reference tests

Some GWGLM tests compare pyGWRx output with optional `mgwr` and `spglm`. These packages are numerical references only; normal pyGWRx GWGLM fitting does not call them.

```bash
python -m pip install -e ".[test,reference]"
python -m pytest -q -m reference
```

## Coverage

The blocking coverage job excludes the slow direct-execution example harness because all examples are exercised separately. It still traces the package with branch coverage and enforces the threshold from `pyproject.toml`.

```bash
python tools/run_coverage.py --batch 1
python tools/run_coverage.py --batch 2
python tools/run_coverage.py --batch 3
python tools/run_coverage.py --combine
```

The batches use separate coverage data files and are combined only after all
three test processes finish. This avoids cumulative tracing slowdowns in the
numerically heavy and plotting portions of the suite while preserving one final
branch-coverage result.

## Installed-distribution smoke test

After installing a wheel, sdist, or TestPyPI release in a clean environment, run the smoke script outside the source tree:

```bash
python tools/smoke_installed_distribution.py
```

It verifies `py.typed`, loads the bundled Columbus data, confirms that no external top-level `mgtwr` package is present, and fits a small internal MGTWR model.
