# HANDOFF — pyGWRx model-handbook research and rewrite

Date: 2026-07-20

## 1. Authoritative repository state

- Repository: `hujinghaoabcd/pyGWRx`
- Active continuation branch: `docs/model-handbook-research-audit`
- Branch base: current `main` at merge commit `e04d88d9668509c1bf72e06c933036c8c25dfd29`
- Package released on PyPI: `pyGWRx 0.1.2`
- This file is the authoritative handoff for the next conversation.

Do not reconstruct the project from old archives, earlier handoff text, isolated model files, or pre-release branches. Continue from the active branch above.

## 2. Important completed changes that must be preserved

### Base-class hierarchy

The previous handoff was obsolete on this point. The current code has already been refactored:

- The former `BaseGWR` implementation was merged into `BaseSpatialRegressor`.
- GWR-family models now use `BaseSpatialRegressor` directly or through the spatiotemporal/multiscale base classes.
- `BaseGWR = BaseSpatialRegressor` remains only as a backward-compatibility identity alias for the 0.1.x series.
- New source code, examples, and documentation should use `BaseSpatialRegressor`, not `BaseGWR`.
- Do not recreate a separate `BaseGWR` inheritance layer.

### MGTWR boundary

- `MGTWR` is self-contained.
- Do not add `mgtwr==2.0.5` or any external MGTWR runtime, optional, development, test, CI, documentation, distribution, or SBOM dependency.
- Do not replace the frozen numerical comparison fixture with a live external import.
- MGWR and MGTWR independent-target prediction remain intentionally unsupported until a validated operator exists.

### MkDocs presentation

The current documentation style has already been changed to a rectangular interface:

- `docs/assets/css/rectangular.css`
- `docs/assets/css/rectangular-final.css`
- `overrides/main.html`
- `overrides/partials/header.html`

The homepage bottom dark CTA is hidden. The English/Chinese header selector is temporarily hidden while `extra.alternate` remains configured. Preserve these choices unless the user explicitly changes them.

## 3. Current user request

The user believes the model documentation is not usable enough for people who do not already understand each algorithm. Every supported model needs a genuine user manual explaining:

- what problem the model solves;
- when it should and should not be used;
- what input data it expects;
- how to use it in pyGWRx;
- every important constructor, `fit`, `predict`, `transform`, or inference parameter;
- what each parameter means;
- how to select parameter values;
- what happens when a value is too small, too large, or inconsistent;
- what fitted attributes mean;
- how results should be interpreted;
- model-specific diagnostics, limitations, and common failures.

The user explicitly rejected the existing approach because many pages look nearly identical. The next assistant must research each established method from its primary literature and maintained authoritative implementation before rewriting its page.

## 4. Confirmed problems in the existing handbook

The current files under `docs/models/` are not an acceptable final handbook.

### Template duplication

Large generic sections were copied across regression, classification, transformation, descriptive-statistics, and inference models. Examples include the same generic decision table, inspection sequence, diagnostics fallback, and reporting text.

This is scientifically inappropriate because:

- GWR and MGWR are continuous-response regression models;
- GWDA is classification;
- GWPCA is a local multivariate transformation;
- GWSS is local descriptive statistics;
- BootstrapGWR is an inference procedure;
- the spatiotemporal models have different time-data contracts.

A shared page structure is allowed; shared model-specific prose is not.

### User-hostile examples

Several handbook examples import repository-only helpers such as:

```python
from _common import spatial_regression
```

Those examples are useful for repository testing but cannot be copied into an external user project. Handbook examples must run after:

```bash
pip install pygwrx
```

They should import only public packages and create or load their own small data.

### Incorrect installation text

Pages currently show development installation such as:

```bash
pip install -e ".[all]"
```

The ordinary user-facing default must be:

```bash
pip install pygwrx
```

Optional extras may be mentioned only where they are actually needed.

### Fragile hand-written signatures

At least the GWR and MGWR pages contain a missing comma between `sigma2_v1` and `verbose` in the displayed constructor signature. Do not manually reproduce long signatures unless they are mechanically checked. The API page may remain the exact signature source, while the handbook uses verified parameter tables and practical constructor examples.

### Unsupported generic claims

Do not say every model supports the same operations, diagnostics, prediction contract, uncertainty outputs, or validation workflow. Check the current source and tests for each class.

## 5. Required documentation method

Every established model page must be based on four sources of truth:

1. **Primary literature** — the defining paper or book chapter.
2. **Maintained authoritative implementation** — for example `GWmodel`, `mgwr`, or the method authors’ maintained software where available.
3. **Current pyGWRx source** — exact parameters, defaults, supported methods, fitted state, and limitations.
4. **pyGWRx tests and runnable examples** — verified behavior and error boundaries.

Always separate these statements:

- what the published method defines;
- what a reference implementation does;
- what pyGWRx currently implements;
- what pyGWRx deliberately does not implement.

Do not silently claim that a pyGWRx extension is part of the original paper. Do not document a literature feature that the package does not implement.

## 6. Standard page structure

Use the same information architecture for consistency, but write model-specific content in every section.

1. Model purpose in plain language
2. Scientific question it answers
3. When to use it
4. When not to use it
5. Difference from the closest alternatives
6. Published statistical formulation
7. pyGWRx implementation notes and departures
8. Input-data contract with shapes, types, units, and preprocessing
9. Minimal standalone example
10. Realistic workflow example
11. Constructor parameter table
12. `fit()` parameter table
13. `predict()`, `predict_result()`, `transform()`, or inference parameter table as applicable
14. Practical parameter-selection guidance
15. Fitted attributes and their shapes
16. Result interpretation
17. Required diagnostics
18. Common warnings, errors, and remedies
19. Reporting checklist for papers and technical reports
20. Primary references and authoritative software references
21. Links to exact API and maintained repository example

### Parameter-table standard

Do not provide only a dictionary-style definition. Each important parameter should include:

| Field | Required content |
|---|---|
| Parameter | Exact public name |
| Type/default | Verified from current source |
| Meaning | Statistical and computational role |
| How to choose | Practical decision rule |
| Too small/large | Expected consequence where meaningful |
| Constraints | Accepted values and relationships |
| Related outputs | Fitted attributes affected by it |

## 7. Example standard

A handbook example must:

- run in a clean external environment after `pip install pygwrx`;
- avoid `_common` and other repository-private helpers;
- use a deterministic seed;
- use `numpy`/`pandas` or a public bundled dataset;
- show the actual input shapes;
- fit the model with realistic settings;
- inspect model-specific fitted attributes;
- demonstrate prediction only when the class supports it;
- explicitly demonstrate a limitation when prediction is intentionally unsupported;
- remain small enough for documentation builds and user experimentation.

Repository examples in `examples/models/` remain the executable API-coverage suite. The handbook may link to them but should not paste their private-helper imports as the primary user example.

## 8. Model inventory and research status

There are 19 public model classes.

| Model | Current source basis already identified | Handbook status / next action |
|---|---|---|
| `GWR` | Brunsdon, Fotheringham & Charlton (1996); Fotheringham, Brunsdon & Charlton (2002); route-map guidance | Source inspected. Rewrite first as the standard single-bandwidth regression manual. |
| `MGWR` | Fotheringham, Yang & Kang (2017); maintained Python `mgwr` literature/software | Source inspected. Rewrite second; emphasize variable-specific scales, backfitting, inference cost, and no independent-target prediction. |
| `RGWR` | Harris, Fotheringham & Juggins (2010); `GWmodel::gwr.robust` | Source inspected at class/docstring level. Verify automatic versus filtered procedures against primary/reference implementation before writing. |
| `STWR` | Research pending | Identify the exact stage-based STWR method implemented by pyGWRx; do not infer from class name. |
| `GTWR` | Huang, Wu & Barry GTWR literature expected; exact implementation must be verified | Inspect source/tests and verify its row-wise time contract, distance formula, causal filtering, and prediction behavior. |
| `GWGLM` | Nakaya et al. (2005) for geographically weighted Poisson regression; local GLM literature | Source inspected. Write separate Gaussian, Poisson, and Bernoulli guidance, including exposure/offset and family-specific residuals. |
| `GWLasso` | Wheeler (2009); maintained CRAN `GWlasso` workflow | Source inspected. Explain local standardization, local penalty selection, bandwidth selection, active variables, and optional dependency requirements. |
| `MixedGWR` | Fotheringham et al. (2002); `GWmodel::gwr.mixed` partial-regression workflow | Source inspected. Explain global/local variable partitioning, global versus local intercept, and identifiability. |
| `GWPCA` | Harris, Brunsdon & Charlton GWPCA literature; authoritative software to verify | Research and source inspection pending. Treat as transformation, not response prediction. |
| `GWDA` | Brunsdon, Fotheringham & Charlton GWDA literature; authoritative software to verify | Research and source inspection pending. Treat as classification with local probabilities. |
| `GWSS` | Geographically weighted summary statistics literature; `GWmodel` likely reference | Research and source inspection pending. Treat as descriptive statistics, not regression. |
| `ScalableGWR` | Murakami et al. scalable GWR literature expected | Verify the exact approximation implemented, complexity, supported kernels, and prediction contract. |
| `LCRGWR` | Wheeler (2007); `GWmodel::gwr.lcr`; Gollini et al. (2015) | Source inspected. Explain local condition numbers, compensation threshold, local ridge, and distinction between pre/post-penalty condition diagnostics. |
| `BootstrapGWR` | Research pending | Identify exact bootstrap/null procedure, resampling unit, returned p-values, and supported inference claim. |
| `SGWR` | Lessani & Li (2024, 2025) | Source inspected. Verify published geographic/similarity mixing, alpha selection, similarity standardization, and software-paper variants. |
| `SGTWR` | Research pending | Identify the exact geography-time-similarity formulation and source; do not treat it as a trivial SGWR+time extension. |
| `MGTWR` | Wu et al. (2019); self-contained pyGWRx implementation with frozen independent numerical validation | Source inspected. Preserve no external dependency. Explain coefficient-specific spatial bandwidths and taus, unit sensitivity, search limitations, inference cost, and no independent-target prediction. |
| `LGGWR` | Original pyGWRx research model | Do not invent an external defining paper. Document its mathematical definition, learned latent geometry, implementation, tests, limitations, and controlled comparisons. Mark clearly as an original research model. |
| `GRGWR` | Original pyGWRx research model | Do not invent an external defining paper. Document regimes, connectivity/assignment logic, implementation, tests, limitations, and controlled comparisons. Mark clearly as an original research model. |

The literature names in this table are starting points, not permission to write from memory. Verify bibliographic metadata and method details from primary sources before final prose.

## 9. Source findings already established

### GWR

Current class: `src/pygwrx/models/gwr.py`

Constructor parameters verified from source:

- `kernel="gaussian"`
- `bandwidth="cv"`
- `bandwidth_method="cv"`
- `adaptive=False`
- `bandwidth_range=None`
- `optimization_method="golden_section"`
- `fit_intercept=True`
- `distance_metric="euclidean"`
- `sigma2_v1=True`
- `verbose=False`

`fit()` parameters verified from source:

- `X`, `y`, `coords`
- `compute_hat_matrix=True`
- `compute_local_r2=True`
- `compute_inference=True`
- compatibility alias `compute_hat_matrix_flag=None`
- optional per-fit `verbose=None`

Important behavior:

- Supports fixed-distance and adaptive-neighbour bandwidths.
- Automatic criteria accepted by GWR are `cv`, `aic`, `aicc`, and `bic`.
- Adaptive numeric bandwidths must be integer neighbour counts.
- New-location `predict()` and `predict_result()` recalibrate local coefficients from stored training data; they do not simply interpolate calibration coefficients.
- `predict_result()` can expose predictions, local coefficients, intercepts, standard errors, and t values when inference was enabled.
- `compute_hat_matrix=False` avoids storing the full matrix while traces and influence remain available.
- `sigma2_v1` selects between two residual-variance denominators.

The current `docs/models/gwr.md` must be replaced rather than lightly edited.

### MGWR

Current class: `src/pygwrx/models/mgwr.py`

Constructor parameters verified from source:

- `kernel="bisquare"`
- `bandwidths=None`
- `bandwidth_method="aicc"`
- `adaptive=True`
- `bandwidth_range=None`
- `bandwidth_ranges=None`
- `init_bandwidth=None`
- `optimization_method="golden_section"`
- `search_tol=1e-6`
- `search_max_iter=200`
- `max_iter=200`
- `tol=1e-5`
- `rss_score=False`
- `bws_same_times=5`
- `fit_intercept=True`
- `distance_metric="euclidean"`
- `sigma2_v1=True`
- `verbose=False`

`fit()` parameters verified from source:

- `X`, `y`, `coords`
- `compute_hat_matrix=False`
- `store_partial_hat_matrices=False`
- `compute_inference=True`
- `n_chunks=1`
- optional per-fit `verbose=None`

Important behavior:

- One bandwidth is estimated for every fitted parameter, including the intercept when enabled.
- A scalar manual `bandwidths` value is repeated for all parameters; a sequence must contain exactly one entry per fitted parameter.
- `bandwidth_ranges` follows the same per-parameter count rule.
- The model uses iterative additive backfitting and records `bandwidth_history_`, `convergence_history_`, `n_iter_`, and `converged_`.
- Exact smoother inference is computationally and memory intensive.
- `n_chunks` reduces the memory used while building exact smoother quantities; it is not a parallelism parameter.
- `store_partial_hat_matrices=True` may require very large memory because it retains one `n x n` smoother per fitted parameter.
- `predict()` intentionally raises `NotImplementedError` for independent targets. Use `fitted_values_` for calibration locations.
- Key outputs include `bandwidths_`, `effective_params_by_variable_`/`ENP_j_`, local parameter surfaces, standard errors/t values, adjusted alpha levels, critical t values, convergence state, and diagnostics.

The current `docs/models/mgwr.md` must be replaced rather than lightly edited.

### Other source files already inspected during this conversation

- `src/pygwrx/models/rgwr.py`
- `src/pygwrx/models/mixed_gwr.py`
- `src/pygwrx/models/gw_lasso.py`
- `src/pygwrx/models/sgwr.py`
- `src/pygwrx/models/lcr_gwr.py`
- `src/pygwrx/models/glm_gwr.py`
- `src/pygwrx/models/mgtwr.py`
- `src/pygwrx/core/kernels.py`
- `src/pygwrx/core/bandwidth.py`

Their class docstrings contain useful implementation facts and starting references, but these have not yet been converted into final handbook pages.

## 10. Immediate next tasks

The previous assistant stated that it would create an audit matrix and rewrite GWR/MGWR, but no handbook page has yet been modified on this branch. The next conversation should perform the following in order.

### Task A — create the research audit

Create:

```text
docs/development/model-handbook-audit.md
```

For all 19 models, record:

- defining literature;
- maintained authoritative implementation;
- pyGWRx source file;
- relevant tests;
- maintained example;
- constructor/fit/prediction contract;
- documentation risks;
- verification status;
- remaining questions.

Add the audit page under the Development section of `mkdocs.yml` only after it has useful content.

### Task B — rewrite GWR as the first complete exemplar

Replace `docs/models/gwr.md` with a source- and literature-grounded manual. It must include:

- standalone user example without `_common`;
- verified constructor and method parameter tables;
- fixed versus adaptive bandwidth explanation with units;
- kernel behavior based on the actual built-in kernels;
- CV/AIC/AICc/BIC selection guidance;
- `sigma2_v1`, inference flags, and hat-matrix memory guidance;
- new-location recalibration explanation;
- fitted attribute shapes and interpretation;
- diagnostics, reporting, and realistic failure remedies.

### Task C — rewrite MGWR as a genuinely different exemplar

Replace `docs/models/mgwr.md` with a model-specific manual. It must include:

- different-scale scientific motivation;
- intercept-inclusive bandwidth ordering;
- manual versus automatic bandwidth rules;
- backfitting and convergence parameters;
- `bandwidth_range` versus `bandwidth_ranges`;
- exact inference and memory controls;
- variable-specific effective parameters and multiple-testing outputs;
- explicit no-independent-target-prediction boundary;
- interpretation cautions: bandwidth is evidence about scale, not a literal causal process radius.

### Task D — validate before moving to the next batch

Do not rewrite all 19 pages in one unreviewed mechanical pass. First validate the GWR and MGWR style and content. Then proceed in model-specific batches, for example:

1. `RGWR`, `LCRGWR`, `GWLasso`, `MixedGWR`
2. `GWGLM`, `GWDA`
3. `GWPCA`, `GWSS`, `BootstrapGWR`
4. `GTWR`, `STWR`, `SGTWR`, `MGTWR`
5. `ScalableGWR`, `SGWR`
6. original models `LGGWR`, `GRGWR`

## 11. Validation and pull-request workflow

Work on `docs/model-handbook-research-audit` or a child branch. Do not write directly to `main`.

Before opening or merging a PR:

```bash
python tools/generate_example_docs.py
python tools/generate_api_docs.py
git diff --exit-code
mkdocs build --strict
```

The generated API and example pages must remain current. The model-handbook pages under `docs/models/` are curated manuals and must not be overwritten by a generic generator.

Run the repository’s standard quality checks and allow GitHub Actions Documentation, CI, and Security workflows to complete. Documentation-only changes still require strict build and generated-document consistency.

## 12. Writing and project conventions

Preserve:

- Google-style docstrings;
- English source comments;
- SPDX and Jinghao Hu source headers;
- explicit fitted-state reset behavior;
- stable public API;
- strict typed public surface;
- current NumPy/SciPy self-contained numerical implementations;
- `mkdocs<2` and `mkdocs-material<10` until a separately validated migration;
- current rectangular documentation style.

For handbook prose:

- write the main English page first;
- update Chinese pages only after the English technical content is verified;
- use plain language before equations;
- define all units and array shapes;
- distinguish prediction from calibration-location fitted values;
- distinguish association from causation;
- never claim benchmark superiority without controlled evidence;
- never present a paper citation as proof that pyGWRx implements every feature in that paper.

## 13. Completion criterion

The documentation overhaul is complete only when a new user can answer, for every model:

1. What does this model do?
2. Is it suitable for my response type and scientific question?
3. What arrays or tables must I provide?
4. Which parameters matter first?
5. What do their values and units mean?
6. How do I run a complete example outside the repository?
7. What outputs should I inspect?
8. What diagnostics are mandatory?
9. What operations are intentionally unsupported?
10. Which primary sources define the method?

Until all ten questions are answered with model-specific, source-verified content, the corresponding handbook page is not finished.
