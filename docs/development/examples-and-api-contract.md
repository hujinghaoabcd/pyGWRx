# Examples and API contract

The supported public API is defined by the `__all__` lists of `pygwrx.models`, `pygwrx.core`, `pygwrx.diagnostics`, `pygwrx.plotting`, and `pygwrx.io`.

## Generators

`tools/generate_api_docs.py`:

1. imports the current public namespaces
2. generates grouped API pages under `docs/api/`
3. adds purpose, import path, signature, full Docstring, maintained example, and embedded source
4. writes `examples/API_COVERAGE.json` and `.csv`

`tools/generate_example_docs.py`:

1. scans the 45 maintained scripts
2. extracts their purpose and public imports
3. documents required extras, run commands, inspection targets, related guides, and full source
4. writes the six category pages under `docs/examples/`

`examples/validate_coverage.py` verifies that every public symbol has a valid mapping and that no stale API entry remains.

```bash
python tools/generate_api_docs.py
python tools/generate_example_docs.py
python examples/validate_coverage.py
```

A new public symbol is incomplete until its Docstring, API group, runnable use, and example mapping all exist. A new example is incomplete until it can be run in isolation and appears in the generated catalog.
