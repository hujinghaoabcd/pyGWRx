# pyGWRx 0.1.2 modification and validation report

Date: 2026-07-19  
Authoritative input: the uploaded pyGWRx 0.1.2 complete-project archive only

## 1. Scope and invariant

This audit addressed the remaining release-review requirements while preserving the existing design decisions. `BaseGWR` was explicitly excluded from modification. Its source SHA-256 before and after the work is:

```text
b9a22952ede2ec57ded6973af8e0385f37d3ee172465f9973d53ac7fa36dfcd8
```

## 2. Self-contained MGTWR

The former external-backend wrapper was replaced by a pyGWRx-native implementation that reuses the package's distance, kernel, weighted least-squares, summary, diagnostics, and fitted-state components. It implements:

- GTWR initialization;
- variable-specific spatial bandwidths and temporal scales;
- additive partial-residual backfitting;
- fixed and adaptive kernel semantics;
- Gaussian, bisquare, and exponential kernels;
- AIC, AICc, BIC, and CV scale criteria;
- convergence, bandwidth, and tau histories;
- optional exact smoother propagation for ENP, influence, covariance factors, standard errors, t statistics, AIC, AICc, and BIC;
- explicit calibration-location-only prediction boundaries.

No external MGTWR runtime, optional, development, test, reference, CI, documentation, wheel, sdist, or SBOM dependency remains.

### Independent numerical comparison

A one-time independent implementation was executed outside the project tree with identical inputs and fixed scales. Seven configurations covered three kernels, fixed/adaptive bandwidth semantics, and intercept/no-intercept fitting. Worst observed absolute differences were:

| Quantity | Maximum absolute difference |
|---|---:|
| Local parameters | `4.17e-8` |
| Fitted values | `7.29e-8` |
| R² | `9.71e-10` |
| ENP by coefficient | `1.67e-6` |
| Parameter standard errors | `4.14e-9` |
| Parameter t values | `6.84e-6` |

Backfitting iteration counts were identical in all configurations. A sanitized fixed-scale fixture is stored in `tests/reference_data/mgtwr_fixed_gaussian_reference.json`; the repository does not retain the comparison implementation.

Automatic-search outputs can differ because pyGWRx explicitly evaluates configured boundaries and uses deterministic coarse-to-fine candidate grids. This is documented as bounded deterministic search rather than an exhaustive global-optimum guarantee.

## 3. Information-criterion correction

The audit found that the shared Gaussian `compute_aic()` and `compute_bic()` omitted parts of the standard maximized-likelihood expression and the additional residual-variance parameter. They now use:

```text
AIC = n log(RSS/n) + n log(2π) + n + 2[trace(S) + 1]
BIC = n log(RSS/n) + n log(2π) + n + [trace(S) + 1] log(n)
```

AICc already matched the standard Gaussian GWR expression. `tests/test_metrics.py` provides exact formula regression tests, and the frozen MGTWR fixture validates all three criteria through the model path.

## 4. API consistency

Inactive or single-choice public parameters were removed from GWPCA, MixedGWR, BootstrapGWR, SGWR, GWLasso, and ScalableGWR. Source signatures, tests, examples, generated API pages, English manuals, Chinese manuals, and release materials are synchronized. Deliberate `NotImplementedError` boundaries for independent-target MGWR/MGTWR prediction remain documented rather than being hidden by an invalid fallback.

## 5. Typing and local quality gates

- `pygwrx/py.typed` is included in distributions.
- Blocking mypy covers the explicitly documented strict typed surface.
- Black, Ruff, isort, and mypy are aligned between CI and pre-commit.
- The obsolete Flake8-only configuration was removed.
- Generated example/API documentation is checked for a clean diff.
- Branch coverage runs in three isolated test-file processes and combines the
  resulting data before applying the unchanged 74% threshold.
- The 45-example runner retains one subprocess per script but uses bounded
  concurrency and reports individual timeouts instead of aborting opaquely.

## 6. CI and release workflows

The project now defines:

- Windows/Linux/macOS × Python 3.11/3.12/3.13/3.14 in main CI;
- the same 12-combination compatibility gate before tag publication;
- separate quality, coverage, optional reference, minimum-dependency, distribution-install, documentation, and security jobs;
- least-privilege `GITHUB_TOKEN` permissions at workflow/job scope;
- explicit job timeouts;
- strict documentation builds;
- TestPyPI and PyPI OIDC Trusted Publishing;
- TestPyPI propagation retries and exact-version installation;
- GitHub Release assets and external SHA-256 manifest.

A workflow definition is not evidence of a successful remote run. Windows/macOS and Python versions other than the local environment must be verified from GitHub Actions before claiming cross-platform success.

## 7. Distribution hardening

`tools/smoke_installed_distribution.py` is run outside the source tree after installing a wheel, sdist, or TestPyPI release. It verifies:

- installed package version/import;
- `py.typed` presence;
- bundled Columbus dataset loading;
- absence of an external top-level `mgtwr` package;
- actual fitting of a small self-contained MGTWR model;
- finite fitted output and expected parameter shape.

The sdist now includes tests, the frozen MGTWR fixture, pre-commit configuration, and the smoke tool. `tools/verify_distributions.py` checks these files and rejects forbidden dependency metadata.

## 8. Security and supply chain

- `SECURITY.md` is present.
- Runtime dependencies are audited online by `pip-audit` in an isolated environment.
- A reproducible CycloneDX JSON SBOM is generated from the installed wheel runtime.
- Trusted Publishing avoids long-lived PyPI credentials.
- SHA-256 values are generated for all release deliverables.

The local execution container could not complete the online vulnerability query because outbound DNS/network access was unavailable. This is recorded as **not completed**, not as a pass. The GitHub security job remains the authoritative online gate.

## 9. Dataset evidence

- Exact upstream package releases or repository commit pins are recorded.
- Evidence date, source object/path, local processing, and integrity status are exposed through dataset metadata.
- `DATA_HASHES.sha256` covers all 24 bundled files.
- Offline verification checks every local SHA-256 and three pinned Git-blob identities.
- Provenance is kept distinct from byte-identity claims.

## 10. Local validation record

Environment: Linux, Python 3.13.

| Gate | Result |
|---|---|
| Black | Passed |
| isort | Passed |
| Ruff E/F/I | Passed |
| Blocking mypy typed surface | Passed |
| Complete pytest suite | Passed: 363 tests |
| Non-reference suite | Passed: 359 tests; 4 reference tests deselected |
| Independent reference suite | Passed: 4 tests; 359 deselected |
| Branch-aware coverage | Passed: 74.5%; threshold 74% |
| Public API/example map | Passed: 174/174 |
| Maintained examples | Passed: 45/45 by direct execution |
| Dataset integrity/provenance | Passed: 24 SHA-256 entries; 3 Git blobs |
| Strict MkDocs build | Passed |
| Local online `pip-audit` | Not completed: outbound DNS unavailable |
| 12 OS/Python combinations | Configured as blocking remote jobs; not locally executed |

Non-reference tests are executed in three isolated, ten-minute-bounded processes to prevent third-party numerical or plotting teardown from stalling a completed suite.

The documentation dependency set now explicitly constrains `mkdocs<2.0` and `mkdocs-material<10.0`. The current Material theme and mkdocstrings plugin stack depends on MkDocs 1.x APIs; a future documentation-platform migration should be evaluated separately rather than accepted through an unconstrained major upgrade.

## 11. Remaining intentional limitations

- MGTWR automatic scale calibration is deterministic and bounded but is not advertised as an exhaustive global optimizer.
- MGTWR `tau` is unit-dependent; coordinate/time scaling must be reported.
- Exact MGTWR inference can be expensive; `n_chunks` controls memory partitioning, not parallel speedup.
- MGWR and MGTWR independent-target prediction remain unavailable until a validated operator is implemented.
- Whole-package strict mypy compliance is not claimed; the typed surface should be expanded incrementally.
- The project remains Alpha and should obtain successful remote CI, TestPyPI, and online security results before production publication.
