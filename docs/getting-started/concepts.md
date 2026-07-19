# Core concepts

## Spatial non-stationarity

A global regression assumes that one coefficient vector describes the entire study region. A geographically weighted model allows coefficients or local statistics to vary by focal location. This is useful when the same predictor has different associations in different geographic contexts.

Local variation is not automatically causal, meaningful, or stable. It can also be produced by noise, uneven sampling, omitted variables, local collinearity, influential observations, or a bandwidth that is too small.

## Local weighted estimation

For standard GWR at location $s_i$,

$$
\widehat{\boldsymbol\beta}(s_i)
=\left(X^\top W_iX\right)^{-1}X^\top W_i y.
$$

The only difference from ordinary weighted least squares is that every focal location has a different diagonal weight matrix $W_i$. Because local windows overlap, one observation can influence many local fits.

## Kernels

A kernel converts distance into non-negative weight.

- Gaussian: smooth decay with positive weight at all distances.
- Exponential: continuous but sharper decay.
- Bisquare: smooth compact support; zero outside the bandwidth.
- Tricube: another smooth compact kernel.
- Boxcar: equal weight inside the threshold and zero outside.

Kernel choice and bandwidth jointly define the effective local sample.

## Fixed and adaptive bandwidths

- **Fixed bandwidth:** a distance in the chosen coordinate metric. It is useful when a constant physical range is meaningful.
- **Adaptive bandwidth:** an integer neighbour count. The physical radius expands in sparse regions and contracts in dense regions.

The same number has a completely different meaning under `adaptive=False` and `adaptive=True`.

## Distance is part of the model

Euclidean distance on projected coordinates is not equivalent to Euclidean distance on longitude/latitude degrees. Spatiotemporal and similarity-based models add further definitions of proximity:

- GTWR combines spatial and temporal distance.
- STWR uses ordered stages and response-change information.
- SGWR combines geography and attribute similarity.
- SGTWR combines space, time, and similarity.
- LGGWR learns a latent geometry from coordinates and attributes.
- GRGWR introduces connected regime structure.

A neighbourhood definition must be scientifically defensible and available at prediction time.

## Bandwidth selection

CV, AIC, AICc, and BIC are available where implemented. Automatic selection repeatedly fits local models, so it can be expensive. Report:

- criterion;
- search range;
- optimizer or candidate grid;
- tolerance and iteration limits;
- whether the bandwidth is fixed or adaptive;
- whether multiscale variables have separate bandwidths.

A boundary optimum often indicates that the range is too narrow or the model is effectively more global/local than expected.

## Hat matrix, ENP, and model complexity

Many Gaussian local smoothers can be written as $\widehat y=Sy$. The trace of the smoother/hat matrix contributes to the effective number of parameters (ENP). A highly local model can have much greater effective complexity than the number of columns in `X`.

This is why in-sample $R^2$ alone is not a fair model-selection criterion. Use adjusted diagnostics, information criteria, and held-out validation.

## Local inference

Where supported, pyGWRx exposes local standard errors, test statistics, p-values, and significance masks. Important cautions:

- local tests are numerous and spatially dependent;
- multiple-testing adjustment may be needed;
- local collinearity can inflate standard errors and reverse coefficient signs;
- a visually large coefficient may be uncertain;
- bootstrap or specialized inference is not identical across model families.

## Prediction is model-specific

- Regression models may use `predict()` and sometimes `predict_result()`.
- `GWDA` uses `predict()` and `predict_proba()`.
- `GWPCA` uses `transform()`.
- `GWSS` reports local statistics.
- `BootstrapGWR` performs inference.
- `MGWR` and `MGTWR` currently reject independent-target prediction.

Do not silently substitute calibration fitted values for held-out predictions.

## Validation

Random row splits can be misleading when nearby observations share information. Choose validation according to the intended use:

- spatial blocks for geographic transfer;
- forward or rolling splits for forecasting;
- combined space-time blocks for spatiotemporal transfer;
- repeated seeds/restarts for optimization-based research models;
- nested tuning when bandwidths and penalties are selected.

## Interpretation hierarchy

A defensible local-model conclusion should survive the following sequence:

1. global baseline;
2. model selection and bandwidth sensitivity;
3. uncertainty and local collinearity;
4. influence and residual diagnostics;
5. spatial/temporal validation;
6. comparison with a simpler alternative;
7. substantive interpretation with units and limitations.
