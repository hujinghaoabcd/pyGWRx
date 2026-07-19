# pyGWRx 0.1.2 release notes

Release date: 2026-07-19

pyGWRx 0.1.2 is a release-audit and numerical-correctness update. It preserves the uploaded 0.1.2 baseline decisions, leaves `BaseGWR` unchanged, replaces the former MGTWR delegation with a self-contained implementation, and strengthens the package from source validation through isolated distribution installation and OIDC publication.

## MGTWR

`MGTWR` is implemented entirely inside pyGWRx. The model provides coefficient-specific spatial bandwidths and temporal scales, GTWR initialization, additive backfitting, deterministic coarse-to-fine scale selection, convergence histories, fitted diagnostics, and optional exact smoother inference.

A one-time independent comparison was performed outside the repository. With identical fixed bandwidths and temporal scales across Gaussian, bisquare, and exponential kernels, fixed/adaptive bandwidth semantics, and intercept/no-intercept cases, the largest observed differences were:

- local coefficients: `4.17e-8`;
- fitted values: `7.29e-8`;
- coefficient-specific ENP: `1.67e-6`;
- parameter standard errors: `4.14e-9`;
- parameter t statistics: `6.84e-6`.

Backfitting iteration counts matched in all seven comparison configurations. A sanitized fixed-scale result is frozen in `tests/reference_data/mgtwr_fixed_gaussian_reference.json`; normal tests use those values without importing, installing, or declaring an external MGTWR package.

Automatic scale selection remains intentionally documented as a deterministic bounded candidate search with local refinement, not as proof of an exhaustive global optimum. The interpretation of `tau` depends on coordinate and time units. Independent-target prediction remains an explicit capability boundary.

## Statistical diagnostics

The shared Gaussian GWR AIC and BIC formulas were corrected to include the full maximized Gaussian likelihood constant and the residual-variance parameter penalty:

```text
AIC = n log(RSS/n) + n log(2π) + n + 2[trace(S) + 1]
BIC = n log(RSS/n) + n log(2π) + n + [trace(S) + 1] log(n)
```

AICc already used the standard Gaussian GWR form and is retained. Exact regression tests now protect all three criteria.

## Public API cleanup

The release removes constructor parameters that exposed no real choice or always failed:

- `GWPCA.robust`;
- `MixedGWR.auto_select`, `selection_criterion`, and `selection_method`;
- `BootstrapGWR.test_type` and `null_model`;
- `SGWR.similarity_metric`;
- `GWLasso.bandwidth_method`;
- `ScalableGWR.adaptive`.

Source, tests, examples, generated API pages, English manuals, Chinese manuals, and theory documentation are synchronized.

## Typing, CI, and installation

- `pygwrx/py.typed` remains in wheel and sdist.
- Blocking mypy covers a documented strict typed surface rather than making an unsupported whole-package claim.
- Pre-commit now matches the blocking Black, Ruff, isort, and mypy strategy.
- Documentation dependencies explicitly require `mkdocs<2.0` and `mkdocs-material<10.0`; migration to a different documentation engine is deferred to a separately validated change.
- CI and the tag release gate define Windows/Linux/macOS × Python 3.11–3.14 jobs.
- Workflows use least-privilege job permissions, explicit timeouts, current GitHub-maintained action majors, and OIDC Trusted Publishing.
- Wheel, sdist, and TestPyPI verification now run `tools/smoke_installed_distribution.py` outside the source tree. The smoke test loads bundled data, checks `py.typed`, rejects an external top-level `mgtwr` package, and fits the internal MGTWR.
- The sdist includes tests and the frozen MGTWR numerical fixture.

## Security and data evidence

- `SECURITY.md`, blocking online `pip-audit`, and reproducible CycloneDX JSON SBOM generation are included.
- Distribution and SBOM checks reject the removed external MGTWR dependency.
- TestPyPI/PyPI publishing uses GitHub OIDC Trusted Publishing; no long-lived PyPI token is required.
- `DATA_HASHES.sha256` covers all 24 bundled files, with exact package-version or repository-commit provenance and three offline Git-blob identity checks.

## Local validation snapshot

Executed on Linux with Python 3.13:

- `363 passed` in the complete suite;
- `359` non-reference tests;
- `4` independent numerical-reference tests, including the frozen MGTWR fixture;
- `74.5%` branch-aware package coverage, above the configured `74%` threshold;
- `174/174` public symbols mapped to runnable examples;
- `45/45` maintained examples passed by direct execution;
- Black, isort, Ruff, and the blocking mypy typed surface passed;
- strict MkDocs build passed;
- 24 dataset hashes and 3 pinned Git blobs passed offline verification.

The 12 operating-system/Python combinations are configured as blocking remote jobs. They must not be described as passed until the corresponding GitHub Actions run succeeds. Local `pip-audit` could not contact the online vulnerability service in the execution container and is therefore recorded as incomplete, not as a clean result.

## Compatibility note

This remains an Alpha `0.x` release. `BaseGWR` was deliberately excluded from the refactor and retains SHA-256:

```text
b9a22952ede2ec57ded6973af8e0385f37d3ee172465f9973d53ac7fa36dfcd8
```
