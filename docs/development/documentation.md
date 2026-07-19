# Documentation workflow

The public site is partly authored and partly generated. Conceptual model pages, guides, tutorials, and Chinese manuals are maintained as Markdown. API and example catalogs are rebuilt from the current source tree and runnable scripts.

## Local preview

```bash
python -m pip install -e ".[docs]"
python tools/generate_api_docs.py
python tools/generate_example_docs.py
python examples/validate_coverage.py
mkdocs serve
```

Open <http://127.0.0.1:8000>. Changes under `docs/` are reloaded automatically.

## Strict verification

```bash
mkdocs build --strict --clean
```

A strict build must not report missing pages, broken relative links, unresolvable API objects, or navigation omissions.

## Directory responsibilities

- `docs/models/`: detailed English guide for every public model
- `docs/zh/models/`: detailed Chinese guide for every public model
- `docs/guides/`: cross-model numerical, diagnostic, plotting, prediction, temporal, and I/O tasks
- `docs/zh/guides/`: Chinese task guides and API/example navigation
- `docs/tutorials/`: multi-step analyses
- `docs/examples/`: generated catalog of all 45 runnable scripts
- `docs/api/`: generated reference for all 174 public symbols
- `docs/theory/`: algorithm encyclopedia, original-model monographs, and references
- `docs/assets/figures/`: validated gallery figures used by model and plotting guides

Do not manually edit generated files under `docs/api/` or the category pages under `docs/examples/`. Modify the source API, Docstring, runnable example, or generator instead.
