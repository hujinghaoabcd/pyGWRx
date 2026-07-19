# Runnable example suite

The example suite is both user documentation and a release contract. Every public API symbol must appear in at least one concrete script.

| Category | Scripts | Scope |
|---|---:|---|
| Models | 19 | one script for every public model |
| Core | 8 | kernels, distances, solvers, metrics, optimization, bandwidths, and base classes |
| Diagnostics | 5 | model, residual, inference, collinearity, time, weights, and regimes |
| Plotting | 6 | all 56 public plotting functions |
| I/O | 4 | datasets, conversions, persistence, and geospatial round trips |
| Workflows | 3 | end-to-end GWR, model comparison, and space-time comparison |
| **Total** | **45** | **174/174 public symbols** |

## Install the full example environment

```bash
python -m pip install -e ".[all,test]"
```

## Run one example

From the repository root:

```bash
python examples/models/01_gwr.py
```

## Run all examples in isolated subprocesses

```bash
python examples/run_all.py
```

The runner isolates scripts so one Matplotlib state, warning configuration, or optional-model failure does not silently contaminate later examples.

## Verify API coverage

```bash
python tools/generate_api_docs.py
python examples/validate_coverage.py
```

`examples/API_COVERAGE.json` and `.csv` record the namespace, symbol, type, purpose summary, and maintained example path.

## Browse source inline

- [Model examples](models.md)
- [Core examples](core.md)
- [Diagnostics examples](diagnostics.md)
- [Plotting examples](plotting.md)
- [I/O examples](io.md)
- [Workflow examples](workflows.md)

Every page embeds the full source rather than only linking to GitHub.
