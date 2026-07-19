# Contributing

```bash
git clone https://github.com/hujinghaoabcd/pyGWRx.git
cd pyGWRx
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
```

A public change should include implementation, focused tests, a runnable example, regenerated API documentation, and changelog/documentation updates.

```bash
ruff check src tests examples tools
black --check src tests examples tools
python tools/generate_api_docs.py
python examples/validate_coverage.py
pytest -W error
mkdocs build --strict --clean
```

Reuse core validation, kernels, solvers, diagnostics, result conventions, and optional-dependency handling rather than adding isolated model code.
