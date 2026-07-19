# HANDOFF — authoritative pyGWRx 0.1.2 release-audit baseline

Date: 2026-07-19

## Authoritative baseline

The complete project delivered with this handoff is the only continuation baseline for pyGWRx 0.1.2. Do not reconstruct it from the older GitHub tree, 0.1.1, archived notes, or isolated model files.

## Non-negotiable decisions

1. `BaseGWR` was deliberately not renamed or refactored. Its SHA-256 is `b9a22952ede2ec57ded6973af8e0385f37d3ee172465f9973d53ac7fa36dfcd8`.
2. `summary()` returns terminal-friendly plain-text table strings.
3. The inactive `n_jobs` API remains removed.
4. pyGWRx does not claim scikit-learn estimator compatibility.
5. Matplotlib, GeoPandas, and Shapely are base dependencies.
6. `MGTWR` is self-contained. Do not reintroduce any external MGTWR runtime, optional, development, test, reference, CI, documentation, distribution, or SBOM dependency.
7. The frozen MGTWR fixture contains numerical values only. Do not replace it with a live external import.
8. MGWR/MGTWR independent-target prediction boundaries remain explicit until a validated operator is implemented.
9. `py.typed` and the strict typed-surface strategy are retained; do not describe the entire legacy tree as mypy-strict.
10. Keep the current MkDocs-based documentation stack constrained to `mkdocs<2` and `mkdocs-material<10` until a separately validated migration is completed.

## Completed work

- Internal MGTWR implementation with exact smoother inference.
- Real one-time independent fixed-scale comparison and frozen numerical fixture.
- Correct Gaussian GWR AIC/BIC formulas and exact metric regression tests.
- Removal of inactive/single-choice parameters from GWPCA, MixedGWR, BootstrapGWR, SGWR, GWLasso, and ScalableGWR.
- Aligned Black/Ruff/isort/mypy pre-commit and CI quality gates.
- Windows/Linux/macOS × Python 3.11–3.14 main and release matrices.
- Coverage, optional references, minimum dependencies, wheel/sdist installation, strict docs, security, TestPyPI, PyPI, and GitHub Release workflows.
- Actual installed-distribution MGTWR/data smoke testing.
- SECURITY policy, online pip-audit, reproducible CycloneDX SBOM, Trusted Publishing, and release checksums.
- Exact dataset package-version/commit/date/path/processing/hash evidence and offline verification.
- Regenerated API/example documentation and synchronized English/Chinese MGTWR manuals.

## Local validation record

Environment: Linux, Python 3.13.

- Complete suite: 363 passed.
- Non-reference suite: 359 passed, 4 reference tests deselected.
- Independent reference suite: 4 passed, 359 deselected.
- Branch-aware coverage: 74.5%, threshold 74%.
- Non-reference tests are executed in three isolated, ten-minute-bounded processes to prevent third-party numerical or plotting teardown from stalling a completed suite.
- Black, isort, Ruff, and typed-surface mypy: passed.
- API/example coverage: 174/174.
- Data verification: 24 SHA-256 entries and 3 pinned Git blobs passed.
- Strict MkDocs build: passed.
- Maintained examples: 45/45 passed by direct execution.

The local environment cannot prove Windows/macOS or Python 3.11/3.12/3.14 results. The 12 combinations are blocking GitHub Actions jobs and require a successful remote run before a cross-platform pass claim.

The local `pip-audit` invocation could not reach the vulnerability service because the execution container has no outbound DNS/network access. Do not convert this into a clean-audit claim; use the blocking online security workflow.

## MGTWR validation facts

The one-time external comparison was conducted outside the repository and is not part of the distribution. Across seven fixed-scale configurations, worst observed absolute differences were `4.17e-8` for local parameters, `7.29e-8` for fitted values, `1.67e-6` for ENP by coefficient, `4.14e-9` for standard errors, and `6.84e-6` for t values; iteration counts matched.

The implementation's automatic search is a deterministic coarse-to-fine bounded candidate search. Do not describe it as an exhaustive global optimizer. `tau` depends on coordinate and time units. Exact inference is computationally expensive, and `n_chunks` reduces memory rather than adding parallelism.

## Required continuation workflow

Before modifying a model, read its implementation, tests, English manual, Chinese manual, generated API page, and runnable example. Preserve Google-style docstrings, English source comments, SPDX/Jinghao Hu headers, explicit fitted-state reset behavior, and project-standard summaries. For established statistical models, verify mathematical changes against primary literature or maintained official implementations, but do not silently delegate pyGWRx runtime behaviour to an external model package.

After any change, execute the sequence in `docs/development/release.md`, update the validation report and this handoff, rebuild all artifacts, and regenerate external SHA-256 values.
