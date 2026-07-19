# Spatiotemporal data and leakage-safe workflows

pyGWRx has two distinct time contracts: row-wise time and stage-based snapshots. They should not be interchanged.

## Row-wise time: GTWR, SGTWR, MGTWR

```python
model.fit(X, y, coords, times)
```

Each observation has one coordinate and one time. Repeated locations across multiple times are valid.

Document:

- numeric or datetime input;
- time conversion and unit;
- irregular intervals;
- repeated-location structure;
- whether future observations are allowed;
- the target prediction horizon.

## Stage-based time: STWR

```python
model.fit(X_list, y_list, coords_list, intervals)
```

Each list element is a snapshot. STWR uses stage order, intervals, response-change information, and historical-bandwidth evolution. It is not simply GTWR with reshaped arrays.

## Time scaling

Spatial and temporal quantities usually have incompatible units. Model parameters define their interaction:

- GTWR: `lambda_st`, `tau`, `ksi`, and distance-combination choice.
- SGTWR: spatial bandwidth, temporal bandwidth, and `alpha`.
- MGTWR: per-coefficient bandwidths and `taus`.
- STWR: stage intervals, `tick_nums`, `alpha`, and `theta`.

Changing from days to hours can change numerical scale parameters even when the data are otherwise identical. Report the transformation.

## Causal weighting

For forecasting, future observations must not contribute to a focal time. Use `causal=True` where supported and validate the actual weight logic.

A non-causal model can be appropriate for retrospective explanation, but it must not be described as a forecast model.

## Validation designs

### Forward holdout

Train on earlier times and evaluate on later times.

### Rolling origin

Repeatedly expand or move the training window and predict the next period.

### Space-time blocks

Hold out geographic regions and future periods together when the target is transfer to new regions and times.

### Stage holdout

For STWR, reserve the final stage or a sequence of stages and ensure historical lists contain no held-out response information.

## Similarity variables

SGTWR similarity variables must be known at prediction time. Variables constructed from future outcomes, full-period aggregates, or target labels create leakage even when `causal=True`.

## Diagnostic questions

- Are coefficients changing smoothly or only at sparse time groups?
- Does a temporal scale sit at a search boundary?
- Are residuals concentrated in specific periods?
- Does the model outperform a spatial-only GWR on future-safe validation?
- Are apparent improvements caused by future or same-location leakage?

See the [GTWR](../models/gtwr.md), [STWR](../models/stwr.md), [SGTWR](../models/sgtwr.md), and [MGTWR](../models/mgtwr.md) handbooks.
