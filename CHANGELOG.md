# Changelog

All notable changes to pyGWRx are documented here. The format follows Keep a
Changelog and the project uses semantic versioning.

## [Unreleased]

### Removed
- Removed the deprecated public `BaseGWR` compatibility alias. New code should
  inherit from or import `BaseSpatialRegressor`; the consolidated spatial
  regressor hierarchy no longer exposes a duplicate GWR-specific base name.

## [0.1.2] - 2026-07-19

### Changed
- Replaced the external-backend MGTWR wrapper with a self-contained pyGWRx
  implementation using variable-specific spatial bandwidths, temporal scales,
  deterministic joint selection, additive backfitting, and optional exact
  smoother-based inference. Fixed-scale coefficients, fitted values, ENP,
  standard errors, t statistics, and information criteria are now protected by
  a frozen independent numerical fixture.
- Standardized fitted `summary()` methods as terminal-friendly plain-text tables.
- Removed the inactive `n_jobs` constructor parameter and the incomplete
  scikit-learn-style `get_params()` / `set_params()` protocol.
- Removed single-choice or deliberately unsupported constructor parameters from
  GWPCA, MixedGWR, BootstrapGWR, SGWR, GWLasso, and ScalableGWR.
- Documented pyGWRx as a task-specific spatial modelling API rather than a
  scikit-learn estimator compatibility layer.
- Defined a blocking strict typed-surface policy while continuing to distribute
  `py.typed`.
- Constrained the current documentation stack to MkDocs 1.x and Material for
  MkDocs 9.x so dependency resolution cannot silently select incompatible majors.
- Cleaned citation, source-distribution, release, and Trusted Publishing metadata.

### Added
- Added real bundled-data GWR examples covering load, fit, summary, diagnostics,
  prediction, geospatial merge, and export.
- Made every maintained example directly runnable from any working directory.
- Added exact dataset version/commit/date/path/processing records, complete local
  SHA-256 coverage, pinned Git-blob checks, and offline provenance tools.
- Added `SECURITY.md`, blocking `pip-audit`, reproducible CycloneDX SBOM
  generation, and SBOM rejection of the removed external MGTWR package.
- Added Windows/Linux/macOS and Python 3.11-3.14 CI, branch-aware coverage,
  reference tests, oldest-dependency tests, and isolated wheel/sdist installs.
- Added TestPyPI/PyPI OIDC Trusted Publishing, exact TestPyPI install verification,
  and GitHub Release assets for checksums, SBOM, docs, report, and handoff.
- Aligned pre-commit with the blocking Black, Ruff, isort, and typed-API mypy
  gates; added release timeouts and least-privilege workflow permissions.
- Added installed-distribution smoke tests that load bundled data and actually
  fit the self-contained MGTWR from wheel, sdist, and TestPyPI environments.
- Split branch coverage into isolated file batches before combining results,
  preventing cumulative tracing stalls without weakening the final threshold.
- Preserved per-script example isolation while adding bounded concurrent execution
  and explicit timeout reporting to the 45-script validation runner.

### Fixed
- Corrected the shared Gaussian GWR AIC and BIC formulas to include the
  likelihood constant and the residual-variance parameter penalty.
- Corrected the MGTWR paper DOI in the English and Chinese manuals.
- Fixed direct execution failures caused by unresolved `examples/_common.py` imports.
- Corrected bundled spatial-data sidecar and CRS/provenance metadata.
- Synchronized source signatures, tests, generated API pages, examples, English
  manuals, Chinese manuals, and release material after API cleanup.

## [0.1.1] - 2026-07-19

### Changed
- Promoted Matplotlib, GeoPandas, and Shapely to mandatory runtime dependencies.
- Updated installation, quickstart, geospatial I/O, examples, and generated docs.

## [0.1.0] - 2026-07-18

### Added
- Initial alpha release of the standardized pyGWRx model, diagnostics, plotting,
  I/O, example, testing, and documentation suites.
