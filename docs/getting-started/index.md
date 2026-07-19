# Getting started

This section takes a new user from installation to a defensible first analysis. The goal is not merely to make a model run; it is to establish a correct data contract, neighbourhood definition, validation plan, and interpretation workflow.

<div class="grid cards" markdown>

- **[Installation](installation.md)** — Python versions, optional extras, isolated environments, and verification.
- **[Quick start](quickstart.md)** — a complete GWR fit, diagnostics, prediction, and export workflow.
- **[Core concepts](concepts.md)** — local weighting, kernels, bandwidths, hat matrices, inference, and validation.
- **[Data and inputs](data-and-inputs.md)** — array shapes, DataFrames, coordinates, time, classes, exposure, and GeoDataFrames.
- **[Choosing a model](choosing-a-model.md)** — a question-driven decision framework across all 19 models.

</div>

## Recommended first analysis

1. Fit a transparent global baseline.
2. Fit standard GWR with an explicitly documented kernel and bandwidth strategy.
3. Check residuals, influence, local uncertainty, and local collinearity.
4. Use spatially appropriate validation.
5. Add one specialized mechanism at a time—multiscale, robust, temporal, similarity, regularization, or regimes.
6. Report what changed and why the extra complexity is justified.

!!! warning
    A successful `.fit()` call is not evidence that a local model is scientifically appropriate. Local models can make noise look like spatial structure when bandwidths, variables, or validation are poorly chosen.
