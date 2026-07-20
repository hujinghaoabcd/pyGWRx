# Model handbook

pyGWRx exposes 19 public model classes. They share a consistent Python interface where that is scientifically appropriate, but they do **not** solve the same task. This handbook is being rewritten from primary papers, maintained reference implementations, the current pyGWRx source, and regression tests. The goal is to make each page useful to a reader who has never used the model before.

!!! warning "Do not choose a model from its name alone"
    Start from the response type, the scientific question, the assumed neighbourhood, the role of time, and the required inference. A more flexible model is not automatically a better model.

## Start here

| Your question | Start with | Move to another model when... |
|---|---|---|
| Does a continuous-response relationship vary over space? | [`GWR`](gwr.md) | Different predictors appear to operate at different scales: compare [`MGWR`](mgwr.md). |
| Do predictors operate at different spatial scales? | [`MGWR`](mgwr.md) | Some effects should be explicitly global: use [`MixedGWR`](mixed-gwr.md). |
| Are local estimates distorted by response outliers? | [`RGWR`](rgwr.md) | The problem is local predictor collinearity rather than response outliers: use [`LCRGWR`](lcr-gwr.md). |
| Are local coefficients unstable because predictors are correlated? | [`LCRGWR`](lcr-gwr.md) | Local variable selection is the main goal: use [`GWLasso`](gw-lasso.md). |
| Is the response Poisson, Bernoulli, or Gaussian? | [`GWGLM`](gwglm.md) | The response is a class label and class-specific local distributions are required: use [`GWDA`](gwda.md). |
| Do relationships vary in both space and time? | [`GTWR`](gtwr.md) | Data are organised into ordered stages and response-change rates define temporal influence: use [`STWR`](stwr.md). |
| Do coefficients have different spatial and temporal scales? | [`MGTWR`](mgtwr.md) | Attribute similarity should also influence neighbourhoods: compare [`SGTWR`](sgtwr.md). |
| Are geographically distant but contextually similar observations relevant? | [`SGWR`](sgwr.md) | Time is also required: use [`SGTWR`](sgtwr.md). |
| Is the goal local multivariate exploration rather than response prediction? | [`GWSS`](gwss.md) or [`GWPCA`](gwpca.md) | A categorical response is present: use [`GWDA`](gwda.md). |
| Is the sample too large for conventional GWR calibration? | [`ScalableGWR`](scalable-gwr.md) | Exact conventional local regression is still computationally feasible: retain GWR as the reference. |
| Is formal coefficient non-stationarity testing required? | [`BootstrapGWR`](bootstrap-gwr.md) | The goal is prediction rather than a bootstrap test: fit a regression model directly. |
| Should the neighbourhood geometry itself be learned? | [`LGGWR`](lg-gwr.md) | Coefficients are expected to be piecewise smooth with abrupt connected boundaries: use [`GRGWR`](gr-gwr.md). |

## Capability matrix

| Model | Task | Required data | Independent-target operation |
|---|---|---|---|
| [`GWR`](gwr.md) | Single-scale local Gaussian regression | X, y, coordinates | `predict()` and `predict_result()` recalibrate local coefficients at target coordinates. |
| [`MGWR`](mgwr.md) | Multiscale local Gaussian regression | X, y, coordinates | Not exposed; use calibration-location results. |
| [`RGWR`](rgwr.md) | Robust local Gaussian regression | X, y, coordinates | Supported from the fitted robust state. |
| [`STWR`](stwr.md) | Stage-based spatiotemporal regression | Lists of X, y and coordinates by stage, plus intervals | Latest-stage prediction under the fitted historical-stage weighting structure. |
| [`GTWR`](gtwr.md) | Row-wise spatiotemporal regression | X, y, coordinates, row-wise times | Supported at new space-time targets. |
| [`GWGLM`](gwglm.md) | Local Gaussian, Poisson or Bernoulli regression | X, response, coordinates; optional Poisson exposure/offset | Conditional means or probabilities at new targets. |
| [`GWLasso`](gw-lasso.md) | Local penalised regression and variable selection | X, y, coordinates | Supported with fitted scaling and local penalties. |
| [`MixedGWR`](mixed-gwr.md) | Global-local semiparametric regression | X, y, coordinates and variable partition | Supported with global and recalibrated local components. |
| [`GWPCA`](gwpca.md) | Local multivariate transformation | Multivariate X and coordinates | `transform()` returns local component scores; it is not response prediction. |
| [`GWDA`](gwda.md) | Local spatial classification | X, class labels, coordinates | Class labels and local probabilities. |
| [`GWSS`](gwss.md) | Local descriptive statistics | Multivariate X and coordinates | Statistics may be evaluated at summary coordinates; there is no response prediction. |
| [`ScalableGWR`](scalable-gwr.md) | Polynomial-kernel approximation for large samples | X, y, coordinates | Supported through compressed neighbour moments. |
| [`LCRGWR`](lcr-gwr.md) | Local ridge compensation for collinearity | X, y, coordinates | Supported using fitted or locally adjusted ridge terms. |
| [`BootstrapGWR`](bootstrap-gwr.md) | Parametric test of coefficient non-stationarity | X, y, coordinates | Not applicable; this is an inference procedure. |
| [`SGWR`](sgwr.md) | Geographic-plus-attribute-similarity regression | X, y, coordinates and similarity variables | Supported by recomputing both weight components. |
| [`SGTWR`](sgtwr.md) | Space-time-plus-similarity regression | X, y, coordinates, times and similarity variables | Supported at target space-time points. |
| [`MGTWR`](mgtwr.md) | Multiscale spatiotemporal regression | X, y, coordinates and times | Not exposed; use calibration-location results. |
| [`LGGWR`](lg-gwr.md) | Learned latent-neighbourhood regression | X, y, coordinates and context attributes | Supported with the learned geometry. |
| [`GRGWR`](gr-gwr.md) | Connected-regime, piecewise local regression | X, y, coordinates | Supported through target regime assignment and local fitting. |

## Documentation evidence audit

Each model page must distinguish three things:

1. **Published method:** what the cited paper actually defines.
2. **Reference implementation:** behaviour established by maintained author or community software where relevant.
3. **pyGWRx contract:** the parameters, defaults, outputs, extensions and limitations in the current package.

The table below is the rewrite control sheet. “Primary evidence” is not a claim that pyGWRx reproduces every option in that source; model pages must state differences explicitly.

| Model | Primary evidence | pyGWRx correspondence that must be documented | Rewrite status |
|---|---|---|---|
| GWR | Brunsdon, Fotheringham & Charlton (1996), DOI [`10.1111/j.1538-4632.1996.tb00936.x`](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x) | Gaussian local WLS; fixed/adaptive bandwidths; CV/AIC/AICc/BIC; target-location recalibration; optional stored hat matrix. | **Evidence-reviewed manual complete.** |
| MGWR | Fotheringham, Yang & Kang (2017), DOI [`10.1080/24694452.2017.1352480`](https://doi.org/10.1080/24694452.2017.1352480); Oshan et al. (2019), DOI [`10.3390/ijgi8060269`](https://doi.org/10.3390/ijgi8060269) | Additive backfitting; one bandwidth per fitted parameter; exact smoother traces; no independent-target prediction. | **Evidence-reviewed manual complete.** |
| RGWR | Harris, Fotheringham & Juggins (2010), DOI [`10.1080/00045600903550378`](https://doi.org/10.1080/00045600903550378); `GWmodel::gwr.robust` | Iterative automatic downweighting and filtered one-refit modes; robust weights are not ridge penalties. | **Evidence-reviewed manual complete.** |
| STWR | Que et al. (2020), DOI [`10.5194/gmd-13-6149-2020`](https://doi.org/10.5194/gmd-13-6149-2020); STWR v1.0 archive | Ordered stages; response-change-rate temporal effect; `alpha`, `theta` and recent-stage count have model-specific meanings. | **Evidence-reviewed manual complete.** |
| GTWR | Huang, Wu & Barry (2010), DOI [`10.1080/13658810802672469`](https://doi.org/10.1080/13658810802672469); `GWmodel::st.dist` comparison | GWmodel-style distance by default; Euclidean space-time alternative; optional causal filtering is a pyGWRx extension. | **Evidence-reviewed manual complete.** |
| GWGLM | Nakaya et al. (2005), DOI [`10.1002/sim.2129`](https://doi.org/10.1002/sim.2129); maintained GWR/MGWR software conventions | Gaussian, Poisson and Bernoulli families; Poisson exposure/offset; Bernoulli-only binomial contract; local IWLS convergence. | **Evidence-reviewed manual complete.** |
| GWLasso | Wheeler (2009), DOI [`10.1068/a40256`](https://doi.org/10.1068/a40256); current `GWlasso` workflow | Local standardisation; unpenalised intercept; local or fixed alpha; local variable-selection outputs. | **Evidence-reviewed manual complete.** |
| MixedGWR | Brunsdon, Fotheringham & Charlton (1999), DOI [`10.1111/0022-4146.00146`](https://doi.org/10.1111/0022-4146.00146); Mei, He & Fang (2004), DOI [`10.1111/j.1085-9489.2004.00331.x`](https://doi.org/10.1111/j.1085-9489.2004.00331.x) | User-specified global/local partition; partial-regression implementation; intercept can be global or local. | **Evidence-reviewed manual complete.** |
| GWPCA | Harris, Brunsdon & Charlton (2011), DOI [`10.1080/13658816.2011.554838`](https://doi.org/10.1080/13658816.2011.554838); `GWmodel::gwpca` | Basic local weighted SVD; global centring/scaling followed by local centring; optional scores; not a regression model. | **Evidence-reviewed manual complete.** |
| GWDA | Brunsdon, Fotheringham & Charlton (2007), DOI [`10.1111/j.1538-4632.2007.00709.x`](https://doi.org/10.1111/j.1538-4632.2007.00709.x); `GWmodel::gwda` | WLDA/WQDA; local means/covariances/priors; pyGWRx uses the standard Gaussian log-determinant probability formulation. | **Evidence-reviewed manual complete.** |
| GWSS | Brunsdon, Fotheringham & Charlton (2002), DOI [`10.1016/S0198-9715(01)00009-6`](https://doi.org/10.1016/S0198-9715(01)00009-6); `GWmodel::gwss` | Local moment and optional quantile statistics; one shared selected bandwidth; descriptive output rather than prediction. | **Evidence-reviewed manual complete.** |
| ScalableGWR | Murakami et al. (2021), DOI [`10.1080/24694452.2020.1774350`](https://doi.org/10.1080/24694452.2020.1774350) | Published ScaGWR polynomial-kernel estimator; fixed neighbour count Q; optimised scale and global penalty; no full distance matrix. | Source verified; rewrite pending. |
| LCRGWR | Wheeler (2007), DOI [`10.1068/a38325`](https://doi.org/10.1068/a38325); `GWmodel::gwr.lcr` | Local condition-number diagnosis; threshold-triggered ridge compensation; several pre/post-penalty condition-number outputs. | **Evidence-reviewed manual complete.** |
| BootstrapGWR | Harris et al. (2017), DOI [`10.1016/j.spasta.2017.07.006`](https://doi.org/10.1016/j.spasta.2017.07.006); `GWmodel::gwr.bootstrap` | Parametric bootstrap under the MLR null only; coefficient-wise and localised tests; optional bandwidth reselection. | Source verified; rewrite pending. |
| SGWR | Lessani & Li (2024), DOI [`10.1080/13658816.2024.2342319`](https://doi.org/10.1080/13658816.2024.2342319) | Convex combination of geographic and attribute-similarity weights; training-based similarity standardisation; AICc alpha selection. | Source verified; rewrite pending. |
| SGTWR | Li et al. (2025), DOI [`10.3390/su172310773`](https://doi.org/10.3390/su172310773) | Space-time Gaussian component plus SGWR similarity component; deterministic AICc candidate search replaces the paper's genetic algorithm. | Source verified; rewrite pending. |
| MGTWR | Wu et al. (2019), DOI [`10.1080/13658816.2018.1545158`](https://doi.org/10.1080/13658816.2018.1545158) | Variable-specific spatial bandwidths and temporal scales; self-contained additive backfitting; no independent-target prediction. | **Evidence-reviewed manual complete.** |
| LGGWR | **Original pyGWRx research model**; project mathematical specification, implementation, tests and monograph | Learned joint or separable latent geometry; scale-identification constraints; alternating geometry and bandwidth optimisation. It must not be presented as an established external method. | Internal evidence verified; dedicated research-model page pending. |
| GRGWR | **Original pyGWRx research model**; project mathematical specification, implementation, tests and monograph | Connected regime discovery from an initial coefficient field; piecewise-smooth local fitting; conditional AICc excludes discrete search complexity. | Internal evidence verified; dedicated research-model page pending. |

## Rewrite standard for every model page

A completed page must contain model-specific versions of all items below:

- the scientific problem and response type;
- conditions under which the model should and should not be used;
- a comparison with the nearest alternatives;
- a self-contained example that works after `pip install pygwrx` and does not import repository helpers;
- the exact constructor signature from the current source;
- a parameter table explaining meaning, selection strategy and failure modes;
- separate `fit()`, `predict()`, `transform()` or inference-method arguments as applicable;
- fitted attributes and their shapes;
- interpretation rules tied to the model, not generic GWR advice;
- computational and memory implications;
- implementation differences from the paper or reference software;
- common errors and reporting requirements;
- primary references and a link to the generated API page.

## Shared minimum safeguards

These safeguards apply across the family, but they are not a substitute for model-specific guidance:

- establish an appropriate global or simpler baseline first;
- use projected coordinates for ordinary planar distance, or deliberately select a geographic distance metric;
- examine bandwidth boundaries and effective local sample size;
- inspect local collinearity, influential observations and residual spatial structure;
- distinguish exploratory coefficient maps from causal claims;
- use spatially blocked or temporally ordered validation for transfer claims;
- record the exact pyGWRx version and full estimator configuration.

The evidence-reviewed manuals now cover thirteen models: GWR, MGWR, RGWR, STWR, GTWR, GWGLM, GWLasso, MixedGWR, GWPCA, GWDA, GWSS, LCRGWR, and MGTWR. The next batch covers similarity-weighted neighbourhoods.