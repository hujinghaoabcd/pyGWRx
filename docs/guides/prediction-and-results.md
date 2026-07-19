# Prediction and result objects

pyGWRx uses different result semantics for regression, classification, transformation, descriptive statistics, and inference. Do not assume that every class supports `predict()` or that every `predict()` has the same interpretation.

## Calibration results

Where supported:

```python
model.fit(X, y, coords)
calibration = model.to_frame()
```

The calibration table commonly contains coordinates, fitted values, residuals, coefficients, and diagnostics. Preserve an observation ID externally when joining to GIS data.

## Simple prediction

```python
values = model.predict(X_new, coords_new)
```

This returns response values or class labels/probabilities according to the model.

## Rich prediction results

Several regression models provide typed results:

```python
result = model.predict_result(X_new, coords_new)
frame = result.to_frame()
```

Rich results can preserve target coordinates, local parameters, feature names, fitted/predicted values, and optional inference arrays. Use them when downstream mapping or auditing requires more than a one-dimensional prediction array.

## Capability table

| Model | Target operation |
|---|---|
| GWR, RGWR, GTWR, GWGLM, LCRGWR, SGWR, SGTWR, LGGWR, GRGWR, ScalableGWR | `predict()` and/or `predict_result()` |
| GWLasso, MixedGWR | `predict()` |
| GWDA | `predict()` and `predict_proba()` |
| GWPCA | `transform()` |
| MGWR, MGTWR | calibration-location results only |
| GWSS | local statistics only |
| BootstrapGWR | inference only |

## Why MGWR and MGTWR reject independent prediction

Their current validated implementations estimate multiscale coefficient surfaces at calibration locations. The project deliberately raises `NotImplementedError` instead of inventing an unvalidated target-location procedure.

Never catch that exception and use training fitted values as held-out predictions.

## Prediction means local re-calibration

For most local regression models, target prediction is not ordinary interpolation of training coefficient maps. The model forms target-to-training weights and estimates target-local coefficients under its fitted kernel, bandwidth, scaling, similarity, temporal, or latent-geometry state.

## Leakage controls

- Do not use future observations when forecasting.
- Similarity variables and latent attributes must be available at prediction time.
- Scaling parameters must be learned from training data.
- Bandwidth and penalty selection should occur inside the training/tuning procedure.
- Joining target results to geography must use stable IDs and validated cardinality.

## Evaluation

Use metrics appropriate to the task:

- continuous regression: MAE, RMSE, R², calibration and spatial error maps;
- counts: deviance, mean-scale errors, calibration by exposure;
- binary/classification: probability calibration, log loss, AUC where appropriate, class-wise metrics;
- transformation: reconstruction/variance objectives and stability;
- inference: type-I error, power, Monte Carlo uncertainty, and multiplicity handling.

Use spatial blocks or forward-time splits when the intended application requires transfer across space or time.
