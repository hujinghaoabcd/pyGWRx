# pyGWRx active project structure

This document describes the release-facing tree after the example/API finalization pass. Historical development reports are retained under `archive/release-development-notes/` and excluded from release distributions.

```text
pyGWRx/
├── src/pygwrx/
│   ├── __init__.py               public convenience API
│   ├── _optional.py              optional-dependency loader
│   ├── core/                     kernels, distances, solvers, metrics, selectors
│   ├── diagnostics/              uniform diagnostics and inference adapters
│   ├── io/                       datasets, tabular and geographic I/O
│   ├── models/                   19 supported estimators
│   ├── plotting/                 common and model-specific visualizations
│   ├── data/                     currently bundled example datasets
│   └── py.typed                  inline typing marker with documented strict typed-surface policy
├── examples/
│   ├── models/                   one script per supported estimator
│   ├── core/                     every public core symbol
│   ├── diagnostics/              every public diagnostics symbol
│   ├── plotting/                 every public plotting symbol
│   ├── io/                       every public I/O symbol
│   ├── workflows/                integrated workflows
│   ├── API_COVERAGE.csv          human-readable API/example map
│   ├── API_COVERAGE.json         machine-readable API/example map
│   ├── validate_coverage.py      detects missing or stale mappings
│   └── run_all.py                isolated execution of every example
├── docs/
│   ├── getting-started/          installation, concepts, data contracts, model selection
│   ├── models/                   one usage guide per supported model
│   ├── guides/                   diagnostics, plotting, prediction, I/O, time, performance
│   ├── tutorials/                integrated analytical workflows
│   ├── examples/                 rendered catalog of all 45 runnable scripts
│   ├── api/                      generated grouped API pages and inventory
│   ├── development/              testing, contribution, documentation, release
│   ├── project/                  status, stability, citation, licence, archives
│   ├── zh/                       Chinese overview
│   └── assets/                   logo, CSS, and MathJax configuration
├── tools/
│   ├── generate_api_docs.py       reproducible API/coverage generator
│   ├── generate_example_docs.py   reproducible example documentation generator
│   ├── finalize_sbom.py            exact-version SBOM finalizer and dependency guard
│   ├── update_data_hashes.py      deterministic bundled-data hash generator
│   ├── verify_data_provenance.py  offline hash and pinned Git-blob verifier
│   └── verify_distributions.py    wheel/sdist content and metadata gate
├── tests/                        active stable-package tests
├── pyproject.toml                package and dependency metadata
├── mkdocs.yml                    documentation configuration
├── README.md / README.zh.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── DATA_LICENSES.md / THIRD_PARTY_NOTICES.md
├── DATA_PROVENANCE.md / DATA_HASHES.sha256
├── SECURITY.md
├── pyGWRx_0.1.2_release_notes.md
├── MODIFICATION_AND_VALIDATION_REPORT.md
├── HANDOFF_NEXT_CONVERSATION.md
└── archive/release-development-notes/  internal history, excluded from sdist
```

## Public surface

| Namespace | Public symbols | Example coverage |
|---|---:|---:|
| `pygwrx.models` | 26 | 26/26 |
| `pygwrx.core` | 51 | 51/51 |
| `pygwrx.diagnostics` | 23 | 23/23 |
| `pygwrx.plotting` | 56 | 56/56 |
| `pygwrx.io` | 17 | 17/17 |
| **Total** | **173** | **173/173** |

`pygwrx.models` contains 19 estimators and seven public prediction-result classes. Unsupported experimental prototypes and inactive global configuration code are deliberately excluded from this tree.


## Documentation sources

- `docs/models/` and `docs/zh/models/`: 19 detailed model manuals in English and Chinese.
- `docs/guides/` and `docs/zh/guides/`: task-oriented function and workflow guides.
- `docs/api/`: generated reference for all 173 public API symbols.
- `docs/examples/`: generated detailed catalog of all 45 runnable examples.
- `docs/theory/`: algorithm encyclopedia, references, and original-model monographs.
- `tools/generate_api_docs.py`: API and coverage generator.
- `tools/generate_example_docs.py`: detailed example-catalog generator.


## Documentation presentation layer

The public documentation presentation is separated from the Markdown content:

```text
overrides/
└── partials/
    └── announce.html       # optional announcement-bar content

docs/assets/
├── css/extra.css           # complete light/dark visual system
├── images/mark.svg         # compact header and favicon mark
└── js/site.js              # small progressive UI enhancements
```

`mkdocs.yml` keeps the documentation inventory in six task-oriented top-level sections while preserving the complete page tree in the side navigation.
