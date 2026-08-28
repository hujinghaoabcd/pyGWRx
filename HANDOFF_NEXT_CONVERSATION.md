# HANDOFF — pyGWRx continuation baseline and model-handbook rewrite

Date: 2026-07-20

## 1. Authoritative baseline

The GitHub `main` branch of `hujinghaoabcd/pyGWRx` is the only authoritative continuation baseline. At the time of this handoff, the latest merged commit is:

```text
e04d88d9668509c1bf72e06c933036c8c25dfd29
```

Do not reconstruct the project from old ZIP files, archived conversations, PyPI 0.1.1, isolated model files, or stale generated documentation.

The latest public PyPI release is `pyGWRx 0.1.2`. The current documentation and source tree contain unreleased improvements that will belong to a later release.

## 2. Current task

The next task is a complete rewrite of the **Model Handbook** so that a user who does not already understand GWR-family methods can correctly choose, configure, fit, diagnose, and interpret every model.

The current model pages are not an acceptable final baseline. Many were generated from a shared template and therefore repeat nearly identical sections across regression, classification, transformation, descriptive-statistics, inference, and spatiotemporal models. The next conversation must not continue that approach.

The rewrite must be based on all of the following:

1. the primary paper or authoritative methodological source for the model;
2. a maintained official or widely accepted reference implementation when one exists;
3. the current pyGWRx source implementation;
4. the current tests and maintained runnable examples;
5. the actual public API and current limitations.

## 3. Non-negotiable project decisions

1. `BaseSpatialRegressor` is now the core GWR-family regression base class.
2. The former GWR-specific base implementation layer and the deprecated `BaseSpatialRegressor` compatibility alias have been removed.
3. `BaseSpatialRegressor` is the only public GWR-family regression base class.
4. New source, examples, and documentation must use `BaseSpatialRegressor`.
5. `MGTWR` is self-contained. Never reintroduce `mgtwr==2.0.5` or any other external MGTWR runtime, optional, development, test, reference, CI, documentation, distribution, or SBOM dependency.
6. MGWR and MGTWR do not expose independent-target prediction. Do not document such prediction as supported.
7. `summary()` returns terminal-friendly plain-text summaries.
8. pyGWRx does not claim full scikit-learn estimator compatibility.
9. Keep Google-style docstrings, English source comments, SPDX headers, and `Jinghao Hu` author headers.
10. Preserve explicit fitted-state reset behavior and atomic fit failure handling.
11. Keep `py.typed` and the existing typed-public-surface checks.
12. Keep the MkDocs stack constrained to `mkdocs<2` and `mkdocs-material<10` until a separately validated migration is completed.
13. Documentation interface elements use a rectangular style. Do not restore rounded cards or buttons unless the user explicitly changes this decision.
14. The homepage bottom dark CTA block is hidden.
15. The English/Chinese switch is temporarily hidden, while `extra.alternate` remains configured for later restoration.

## 4. Recently completed work

### 4.1 Release and packaging

- `pyGWRx 0.1.2` was released to PyPI.
- Trusted Publishing is configured.
- Cross-platform CI, release, documentation, security, SBOM, wheel/sdist, TestPyPI, PyPI, and GitHub Release workflows are present.
- The README uses a repository-local PyPI version badge because externally generated badge URLs did not render reliably.

### 4.2 Base hierarchy refactor

The old hierarchy used a GWR-specific compatibility layer beneath `BaseSpatialRegressor`.

was consolidated into:

```text
BaseSpatialEstimator
└── BaseSpatialRegressor
    ├── GWR-family models
    ├── BaseSpatiotemporalRegressor
    └── BaseMultiscaleRegressor
```

`BaseSpatialRegressor` contains the kernel, bandwidth, local prediction, and shared GWR-family state from the former GWR-specific base layer. The deprecated compatibility alias has now been removed.

The refactor was merged through PR #6 and passed the repository quality gates.

### 4.3 Documentation style

PR #7 and PR #8 introduced:

- rectangular tables, code blocks, cards, admonitions, API signatures, buttons, search controls, and navigation surfaces;
- a more regular left navigation hierarchy;
- a final CSS override loaded after Material and project styles;
- hidden homepage bottom CTA;
- temporarily hidden language selector.

Do not merge the rectangular CSS files casually. Their loading order is intentional because earlier rules were overridden by Material styles.

## 5. Current handbook problem

The existing pages under `docs/models/` have useful fragments, but most are too template-driven. Known problems include:

- generic decision tables repeated for unrelated model types;
- identical result-reading text copied across many pages;
- development installation commands such as `pip install -e ".[all]"` shown to ordinary users;
- examples importing repository-only helpers such as `_common`;
- automatically rendered signatures containing formatting defects;
- insufficient explanation of how each parameter changes the statistical model;
- insufficient distinction between the original method and pyGWRx-specific implementation choices;
- insufficient explanation of unsupported operations;
- pages that are similar in structure and wording even when the models solve different tasks.

Do not mechanically expand these pages. Rewrite them from verified evidence.

## 6. Supported model inventory

The public model handbook currently covers 19 classes:

1. `GWR`
2. `MGWR`
3. `RGWR`
4. `STWR`
5. `GTWR`
6. `GWGLM`
7. `GWLasso`
8. `MixedGWR`
9. `GWPCA`
10. `GWDA`
11. `GWSS`
12. `ScalableGWR`
13. `LCRGWR`
14. `BootstrapGWR`
15. `SGWR`
16. `SGTWR`
17. `MGTWR`
18. `LGGWR`
19. `GRGWR`

The current capability table is in `docs/models/index.md`, but it must be rechecked against the source before being treated as authoritative.

## 7. Required evidence hierarchy

For every established model, use this order of evidence:

1. original peer-reviewed paper;
2. later methodological paper that defines the implemented variant;
3. official or maintained reference software documentation/source;
4. pyGWRx source code;
5. pyGWRx tests;
6. pyGWRx examples;
7. current generated API page.

When literature and pyGWRx differ, document the difference explicitly.

Use wording such as:

```text
The published method defines ...
pyGWRx implements ...
The current pyGWRx API does not expose ...
```

Never write a paper feature as supported merely because it exists in the literature.

## 8. Sources already confirmed from the current source tree

The following mappings have already been checked in source docstrings and should be verified against the original publications before final writing:

| Model | Current pyGWRx methodological anchor | Important implementation fact |
|---|---|---|
| GWR | Brunsdon, Fotheringham, and Charlton; Fotheringham et al. | Gaussian local weighted least squares; fixed or adaptive bandwidth; new-location recalibration supported. |
| MGWR | Fotheringham, Yang, and Kang (2017) | One bandwidth per fitted parameter; iterative backfitting; exact smoother inference; no independent-target prediction. |
| RGWR | Harris, Fotheringham, and Juggins (2010); `GWmodel::gwr.robust` | Supports `automatic` residual reweighting and `filtered` one-refit procedures. |
| GWGLM | Nakaya et al. (2005) and related GWR GLM work | Supports Gaussian, Poisson, and Bernoulli families; Poisson exposure/offset; Bernoulli only, not grouped binomial. |
| GWLasso | Wheeler (2009); current CRAN `GWlasso` workflow | Local Lasso, locally standardised predictors, unpenalised intercept, optional local CV penalty. |
| MixedGWR | GWR mixed/global-local coefficient literature; `GWmodel::gwr.mixed` | Uses a partial-regression algorithm; variables are partitioned into global and local groups. |
| LCRGWR | Wheeler (2007); `GWmodel::gwr.lcr` | Applies local ridge compensation when condition numbers exceed a threshold. |
| SGWR | Lessani and Li (2024/2025) | Mixes geographic and attribute-similarity weights using `alpha`. |
| MGTWR | Wu et al. (2019) | Variable-specific spatial bandwidths and temporal scale parameters; self-contained backfitting; no independent-target prediction. |
| LGGWR | Original pyGWRx research model | Must be documented from its mathematical definition, source, tests, and controlled experiments, not assigned an unrelated external paper. |
| GRGWR | Original pyGWRx research model | Must be documented as an original research model with explicit implementation and validation boundaries. |

The literature mapping for `STWR`, `GTWR`, `GWPCA`, `GWDA`, `GWSS`, `ScalableGWR`, `BootstrapGWR`, and `SGTWR` still requires a fresh primary-source audit before rewriting their pages.

## 9. Required page structure

Every model page must have the same level of completeness, but not the same prose. Use model-specific sections:

1. **Purpose in plain language**
2. **Scientific question the model answers**
3. **When to use it**
4. **When not to use it**
5. **How it differs from the closest alternatives**
6. **Mathematical model**
7. **How pyGWRx implements the method**
8. **Required input data**
9. **Coordinate, time, class, exposure, stage, or attribute-similarity requirements**
10. **Minimal standalone example**
11. **Complete practical workflow**
12. **Constructor parameters**
13. **`fit()` parameters**
14. **`predict()`, `predict_result()`, `transform()`, `predict_proba()`, or other operation parameters as appropriate**
15. **How to choose each important parameter**
16. **What happens when a parameter is too small, too large, or inappropriate**
17. **Fitted attributes and their shapes**
18. **How to interpret results**
19. **Required diagnostics**
20. **Common warnings and errors**
21. **Current limitations**
22. **What to report in a paper**
23. **Primary references and reference implementations**
24. **Related pyGWRx pages**

Do not add irrelevant sections merely to preserve a template. For example, `GWPCA` requires `transform()` and local component interpretation, not regression prediction sections. `GWDA` requires probability and classification sections. `GWSS` requires local descriptive-statistics interpretation rather than coefficient inference.

## 10. Parameter-table standard

A parameter table must do more than repeat a type annotation. Use at least these columns:

| Parameter | Accepted values/default | Statistical meaning | How to choose | Failure mode or caution |
|---|---|---|---|---|

For example, a bandwidth parameter should explain:

- whether its unit is distance or neighbour count;
- the effect of small and large values;
- whether automatic selection is available;
- which criterion is used;
- the minimum estimable local sample size;
- how sampling-density variation affects the choice;
- whether search boundaries can force a misleading solution.

Boolean and computational parameters also require real guidance. Examples:

- `compute_hat_matrix=False` reduces storage but retains GWR smoother traces in the current implementation;
- MGWR `n_chunks` reduces memory during exact inference and does not create parallel workers;
- `store_partial_hat_matrices=True` can require very large memory;
- `sigma2_v1` changes the residual-variance denominator and therefore the inference values;
- `adaptive=True` changes the bandwidth unit from coordinate distance to neighbour count.

## 11. Example standard

Examples shown in the handbook must be copyable by an installed-package user.

Use:

```bash
pip install pygwrx
```

or, only when an optional dependency is genuinely needed:

```bash
pip install "pygwrx[ml]"
```

Do not use the developer command as the primary user installation instruction:

```bash
pip install -e ".[all]"
```

Handbook examples must not import repository-only helpers:

```python
from _common import ...
```

A documentation example should create or load its own small data, import only public package objects, fit the model, inspect the important outputs, and demonstrate the model-specific operation.

The maintained examples under `examples/` may continue using `_common.py` for repository testing. The handbook example and maintained example may therefore differ, provided both are correct.

## 12. GWR rewrite requirements

The current `docs/models/gwr.md` should be the first rewritten page.

It must explain at least:

- local weighted least squares;
- why GWR differs from OLS;
- fixed-distance versus adaptive-neighbour bandwidths;
- the five built-in kernels: Gaussian, bisquare, exponential, tricube, and boxcar;
- `bandwidth`, `bandwidth_method`, `bandwidth_range`, and `optimization_method`;
- `fit_intercept` and coordinate distance units;
- `sigma2_v1` and its two residual-variance denominators;
- `compute_hat_matrix`, `compute_local_r2`, and `compute_inference`;
- local coefficients, standard errors, t values, local R², influence, standardised residuals, and Cook's distance;
- why local coefficients are associations rather than automatic causal effects;
- new-location prediction: pyGWRx recalibrates the local regression at target coordinates using the training data rather than interpolating stored coefficient surfaces;
- memory implications of the full `n × n` hat matrix;
- minimum local positive-weight observations and the ridge fallback warning;
- projected coordinates versus longitude/latitude and the use of `haversine` when appropriate.

Correct constructor signature:

```python
GWR(
    kernel="gaussian",
    bandwidth="cv",
    bandwidth_method="cv",
    adaptive=False,
    bandwidth_range=None,
    optimization_method="golden_section",
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    verbose=False,
)
```

Current `fit()` controls:

```python
model.fit(
    X,
    y,
    coords,
    compute_hat_matrix=True,
    compute_local_r2=True,
    compute_inference=True,
    verbose=None,
)
```

`compute_hat_matrix_flag` is a compatibility alias and should not be promoted in new examples.

## 13. MGWR rewrite requirements

The second rewritten page should be `docs/models/mgwr.md` and must not read like GWR with a renamed heading.

It must explain:

- why one shared GWR bandwidth can be scientifically restrictive;
- the additive coefficient-specific scale formulation;
- backfitting using partial residuals;
- one bandwidth for every fitted parameter, including the intercept when enabled;
- interpretation of small versus large coefficient-specific bandwidths;
- why a bandwidth is not automatically an exact physical process radius;
- `bandwidths`, `bandwidth_range`, and `bandwidth_ranges`;
- `init_bandwidth`;
- `search_tol`, `search_max_iter`, `max_iter`, `tol`, `rss_score`, and `bws_same_times`;
- convergence history and boundary solutions;
- `effective_params_by_variable_` / `ENP_j_`;
- adjusted alpha and variable-specific critical t values;
- exact smoother inference and its computational cost;
- `n_chunks` as a memory control;
- the high memory cost of `store_partial_hat_matrices=True`;
- the explicit absence of independent-target prediction.

Correct constructor signature:

```python
MGWR(
    kernel="bisquare",
    bandwidths=None,
    bandwidth_method="aicc",
    adaptive=True,
    bandwidth_range=None,
    bandwidth_ranges=None,
    init_bandwidth=None,
    optimization_method="golden_section",
    search_tol=1e-6,
    search_max_iter=200,
    max_iter=200,
    tol=1e-5,
    rss_score=False,
    bws_same_times=5,
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    verbose=False,
)
```

Current `fit()` controls:

```python
model.fit(
    X,
    y,
    coords,
    compute_hat_matrix=False,
    store_partial_hat_matrices=False,
    compute_inference=True,
    n_chunks=1,
    verbose=None,
)
```

Do not show or imply a working independent-target `predict()` workflow. The current method intentionally raises `NotImplementedError`.

## 14. Recommended rewrite order

Use small, reviewable batches rather than rewriting all 19 pages in one unreviewable commit.

### Batch 1 — baseline and multiscale

1. GWR
2. MGWR
3. model evidence/audit table

### Batch 2 — established regression variants

4. RGWR
5. LCRGWR
6. MixedGWR
7. GWLasso
8. GWGLM

### Batch 3 — time and alternative neighbourhoods

9. GTWR
10. STWR
11. SGWR
12. SGTWR
13. MGTWR

### Batch 4 — non-regression and scalable/inference models

14. GWPCA
15. GWDA
16. GWSS
17. ScalableGWR
18. BootstrapGWR

### Batch 5 — original research models

19. LGGWR
20. GRGWR

The count is 20 tasks because the evidence/audit table is an additional deliverable; the model count remains 19.

## 15. English and Chinese documentation policy

Write and validate the English page first. After its technical content is stable:

1. update the corresponding Chinese page;
2. preserve technical equivalence between languages;
3. do not create a shorter or less complete Chinese version;
4. keep parameter names and API identifiers in code formatting;
5. translate explanations, not code identifiers.

The language selector is currently hidden, but the Chinese documentation remains part of the project and must not become stale.

## 16. Research policy

For established statistical models, search the web and use primary sources. Prefer:

- original journal articles;
- official package manuals;
- maintained source repositories;
- peer-reviewed implementation papers.

Avoid relying on unsourced blogs, copied summaries, or a single secondary webpage.

Record for each model:

- full citation;
- DOI or stable publication identifier;
- official/reference implementation;
- exact algorithmic variant implemented by pyGWRx;
- known differences;
- unresolved questions.

For `LGGWR` and `GRGWR`, do not fabricate an external origin. Mark them clearly as original pyGWRx research models and ground the pages in the package's mathematical specification, code, tests, and experiments.

## 17. Files to inspect before rewriting a model

For each model, inspect all of the following:

```text
src/pygwrx/models/<model>.py
tests/ relevant model tests
examples/models/<model example>.py
docs/models/<model>.md
docs/zh/models/<model>.md
docs/api/models/<model>.md
docs/models/index.md
```

Also inspect shared components when relevant:

```text
src/pygwrx/core/base.py
src/pygwrx/core/kernels.py
src/pygwrx/core/bandwidth.py
src/pygwrx/core/optimization.py
src/pygwrx/core/solver.py
src/pygwrx/core/metrics.py
docs/guides/kernels-and-bandwidths.md
docs/guides/diagnostics.md
docs/guides/prediction-and-results.md
```

## 18. Generated documentation warning

`docs/api/` and parts of the example documentation are generated by tooling. Do not hand-edit generated output without also updating its source or generator.

Relevant tools:

```text
tools/generate_api_docs.py
tools/generate_example_docs.py
```

The quality gate checks that generated documentation is current. After source, example, or generator changes, run both generators and commit the resulting documentation.

The hand-written model handbook under `docs/models/` is not a substitute for the API reference, and the generated API reference is not a substitute for a user manual.

## 19. Validation workflow

At minimum, complete these checks before merging a documentation batch:

```bash
python tools/generate_example_docs.py
python tools/generate_api_docs.py
python -m mkdocs build --strict
python -m black --check src tests examples tools
python -m isort --check-only src tests examples tools
python -m ruff check src tests examples tools
```

When source or examples change, also run the relevant tests and maintained examples. Before a release, follow the full process in:

```text
docs/development/release.md
```

Do not claim Windows, macOS, or untested Python-version success from a local Linux run. Use GitHub Actions as the source of truth for the complete matrix.

## 20. Git workflow and current branch

A branch has already been created for the handbook audit:

```text
docs/model-handbook-research-audit
```

At handoff time, no model-handbook rewrite has yet been committed to that branch. It can be used for Batch 1, or replaced by a newly named branch if the repository state makes that safer.

Use a pull request for each coherent batch. Wait for at least:

- strict documentation build;
- quality gates;
- security workflow;
- relevant tests.

Do not mix statistical source changes with large handbook rewrites unless the documentation audit discovers an actual implementation defect that must be fixed and tested.

## 21. First actions for the next conversation

1. Read this file completely.
2. Fetch current `main` and verify the latest commit.
3. Inspect the current `docs/model-handbook-research-audit` branch before writing.
4. Build a model evidence table with one row per model.
5. Search and verify the primary GWR and MGWR sources.
6. Inspect GWR and MGWR source, tests, examples, English pages, Chinese pages, and generated API pages.
7. Rewrite `docs/models/gwr.md` as a standalone user manual.
8. Rewrite `docs/models/mgwr.md` as a distinct multiscale manual.
9. Add independent copy-paste examples that use only public pyGWRx APIs.
10. Update the corresponding Chinese pages after the English content is stable.
11. Run the documentation generators and strict build.
12. Open a focused PR and inspect CI before merging.

## 22. Definition of done for a model page

A page is complete only when a new user can answer all of these without reading the source code:

- What problem does this model solve?
- Is this model appropriate for my response and spatial/temporal structure?
- What shape and meaning must every input have?
- Which parameters do I normally change?
- What does each parameter change statistically?
- How should I choose a bandwidth, scale, penalty, family, or threshold?
- What happens when the chosen value is inappropriate?
- Which operations are supported?
- Which operations are intentionally unsupported?
- What result attributes should I inspect?
- How do I diagnose instability or misspecification?
- How do I interpret the output without making causal overclaims?
- What must I report for reproducibility?
- Which publication and reference implementation define the method?
- How does pyGWRx differ from that reference?

If the page cannot answer these questions, it is not finished.
