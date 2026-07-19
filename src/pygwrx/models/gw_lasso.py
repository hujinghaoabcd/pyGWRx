# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Geographically weighted Lasso regression.

The implementation follows the local coefficient-penalisation framework of
Wheeler (2009) and the current CRAN ``GWlasso`` workflow: spatial weights define
one local regression problem per evaluation site, predictors are standardised
inside each local problem, the intercept is not penalised, and a local Lasso
penalty can be selected by cross-validation.  A global fixed-distance or
adaptive-neighbour bandwidth can be selected by leave-one-out prediction error.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx._optional import import_optional_dependency
from pygwrx.core.base import BaseSpatialRegressor
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import compute_distance_matrix

AlphaLike = Union[float, str]


@dataclass(frozen=True)
class GWLassoPredictionResult:
    """Local GWL coefficient and prediction results at evaluation locations."""

    predictions: Optional[np.ndarray]
    coefficients: np.ndarray
    intercepts: np.ndarray
    alphas: np.ndarray
    active_variables: Tuple[np.ndarray, ...]
    coords: np.ndarray
    feature_names: Tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        """Convert the local result to a pandas data frame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "intercept": self.intercepts,
            "alpha": self.alphas,
        }
        if self.predictions is not None:
            data["prediction"] = self.predictions
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coefficients[:, index]
            data[f"selected_{name}"] = (
                np.abs(self.coefficients[:, index]) > 0.0
            ).astype(int)
        return pd.DataFrame(data)


@dataclass(frozen=True)
class _LocalFit:
    """Internal weighted-Lasso fit result."""

    coefficients: np.ndarray
    intercept: float
    alpha: float
    objective: float
    n_iter: int
    converged: bool
    active: np.ndarray
    cv_score: float


class GWLasso(BaseSpatialRegressor):
    r"""Geographically weighted Lasso regression.

    At evaluation location :math:`s`, the model solves

    .. math::

        \frac{1}{2\sum_i w_i(s)}\sum_i w_i(s)
        \left(y_i-\beta_0(s)-x_i^T\beta(s)\right)^2
        + \lambda(s)\|\beta^*(s)\|_1,

    where :math:`\beta^*` denotes coefficients on locally standardised
    predictors.  The intercept is never penalised.  ``alpha="cv"`` selects a
    separate local penalty at every calibration or prediction location.

    Args:
        kernel: Spatial kernel name or callable. Wheeler's original implementation
            used an exponential kernel; all standard pyGWRx kernels are supported.
        bandwidth: Fixed distance, adaptive neighbour count, or a selection token.
            Use ``"cv"`` to select according to ``adaptive``. ``"adaptive"`` is a
            convenience token that selects an adaptive-neighbour bandwidth by CV.
        alpha: Non-negative fixed Lasso penalty, or ``"cv"`` for a locally selected
            penalty. ``alpha=0`` gives locally weighted least squares after the same
            weighting and intercept conventions.
        alpha_grid: Optional descending or ascending positive penalty candidates.
            When omitted, a local logarithmic path is generated from ``alpha_max``.
        n_alphas: Number of generated local penalty candidates.
        alpha_min_ratio: Smallest generated penalty as a fraction of ``alpha_max``.
        cv_folds: Number of deterministic shuffled folds for local penalty selection.
        standardize: Standardise predictors using local weighted means and scales.
        adaptive: Interpret a numeric bandwidth as an integer neighbour count.
        bandwidth_range: Optional lower and upper bounds for bandwidth selection.
        n_bandwidths: Number of grid candidates used for bandwidth CV.
        max_iter: Maximum coordinate-descent iterations for every local Lasso.
        tol: Coordinate-descent convergence tolerance.
        active_tol: Absolute coefficient threshold used for local variable selection.
        fit_intercept: Estimate an unpenalised local intercept.
        distance_metric: Distance metric used by pyGWRx.
        random_state: Seed used for reproducible local CV folds.
        verbose: Print bandwidth and fitting progress.

    Attributes:
        coef_: Local coefficient matrix with shape ``(n_samples, n_features)``.
        intercept_: Local intercept vector.
        alpha_: Locally selected penalty values.
        active_vars_: Active predictor indices at every location.
        selection_frequency_: Fraction of locations selecting each predictor.
        bandwidth_: Selected fixed distance or adaptive neighbour count.

    References:
        Wheeler, D. C. (2009). Simultaneous coefficient penalization and model
        selection in geographically weighted regression: The geographically
        weighted lasso. *Environment and Planning A*, 41(3), 722-742.

        Mulot, M., & Erb, S. (2025). ``GWlasso``: Geographically Weighted Lasso.
        CRAN package version 1.0.2.
    """

    _ALPHA_TOKEN = "cv"

    def __init__(
        self,
        kernel: Union[str, Callable] = "exponential",
        bandwidth: Union[float, int, str, None] = "cv",
        alpha: AlphaLike = "cv",
        alpha_grid: Optional[Sequence[float]] = None,
        n_alphas: int = 30,
        alpha_min_ratio: float = 1e-3,
        cv_folds: int = 5,
        standardize: bool = True,
        adaptive: bool = False,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        n_bandwidths: int = 8,
        max_iter: int = 5000,
        tol: float = 1e-6,
        active_tol: float = 1e-8,
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        random_state: Optional[int] = 0,
        verbose: bool = False,
    ) -> None:
        adaptive_effective = bool(adaptive)
        if isinstance(bandwidth, str) and bandwidth.strip().lower() == "adaptive":
            adaptive_effective = True

        super().__init__(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method="cv",
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            adaptive=adaptive_effective,
            bandwidth_range=bandwidth_range,
            optimization_method="grid",
            random_state=random_state,
            verbose=verbose,
        )
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.cv_folds = cv_folds
        self.standardize = standardize
        self.n_bandwidths = n_bandwidths
        self.max_iter = max_iter
        self.tol = tol
        self.active_tol = active_tol
        self._validate_lasso_parameters()
        self._reset_gw_lasso_state()

    def _reset_gw_lasso_state(self) -> None:
        """Clear model-specific fitted state."""
        self._mark_unfitted()
        self._reset_regression_state()
        self._reset_gwr_state()
        self.alpha_: Optional[np.ndarray] = None
        self.local_alpha_cv_score_: Optional[np.ndarray] = None
        self.active_vars_: Optional[List[np.ndarray]] = None
        self.selection_frequency_: Optional[np.ndarray] = None
        self.local_objective_: Optional[np.ndarray] = None
        self.n_iter_: Optional[np.ndarray] = None
        self.converged_: Optional[np.ndarray] = None
        self.bandwidth_scores_: Optional[pd.DataFrame] = None
        self.mean_active_variables_: Optional[float] = None
        self.parameter_names_: Optional[Tuple[str, ...]] = None

    def _validate_lasso_parameters(self) -> None:
        """Validate GWL-specific constructor parameters."""
        if isinstance(self.alpha, str):
            if self.alpha.strip().lower() != self._ALPHA_TOKEN:
                raise ValueError("alpha must be non-negative or the string 'cv'.")
        else:
            if isinstance(self.alpha, (bool, np.bool_)):
                raise TypeError("alpha must be non-negative or the string 'cv'.")
            alpha_value = float(self.alpha)
            if not np.isfinite(alpha_value) or alpha_value < 0:
                raise ValueError("alpha must be finite and non-negative.")

        if self.alpha_grid is not None:
            grid = np.asarray(self.alpha_grid, dtype=float).reshape(-1)
            if grid.size == 0 or not np.all(np.isfinite(grid)):
                raise ValueError("alpha_grid must contain finite positive values.")
            if np.any(grid <= 0):
                raise ValueError("alpha_grid values must be greater than zero.")

        if not isinstance(self.n_alphas, (int, np.integer)) or int(self.n_alphas) < 2:
            raise ValueError("n_alphas must be an integer >= 2.")
        if (
            not np.isfinite(self.alpha_min_ratio)
            or self.alpha_min_ratio <= 0
            or self.alpha_min_ratio >= 1
        ):
            raise ValueError("alpha_min_ratio must satisfy 0 < value < 1.")
        if not isinstance(self.cv_folds, (int, np.integer)) or int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be an integer >= 2.")
        if not isinstance(self.standardize, (bool, np.bool_)):
            raise TypeError("standardize must be boolean.")
        if (
            not isinstance(self.n_bandwidths, (int, np.integer))
            or int(self.n_bandwidths) < 2
        ):
            raise ValueError("n_bandwidths must be an integer >= 2.")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be finite and positive.")
        if not np.isfinite(self.active_tol) or self.active_tol < 0:
            raise ValueError("active_tol must be finite and non-negative.")

    @property
    def _alpha_is_cv(self) -> bool:
        return isinstance(self.alpha, str)

    def _feature_names(self) -> Tuple[str, ...]:
        if self.feature_names_in_ is None:
            return tuple(f"x{index}" for index in range(int(self.n_features_in_ or 0)))
        return tuple(str(name) for name in self.feature_names_in_)

    def _weights(
        self, distances: np.ndarray, bandwidth: Union[float, int]
    ) -> np.ndarray:
        """Return valid fixed or adaptive spatial weights."""
        if self.kernel_func_ is None:
            raise RuntimeError("kernel_func_ is not initialised.")
        distances_arr = np.asarray(distances, dtype=float).reshape(-1)
        if self.adaptive:
            distance_bandwidth = adaptive_bandwidth_weights(
                distances_arr,
                int(bandwidth),
            )
        else:
            distance_bandwidth = float(bandwidth)
        weights = np.asarray(
            self.kernel_func_(distances_arr, distance_bandwidth),
            dtype=float,
        ).reshape(-1)
        weights[~np.isfinite(weights)] = 0.0
        weights[weights < 0.0] = 0.0
        return weights

    @staticmethod
    def _weighted_moments(
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        *,
        fit_intercept: bool,
        standardize: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Create a locally centred and scaled weighted-Lasso problem."""
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("Local spatial weights must have a positive sum.")

        if fit_intercept:
            x_mean = np.average(X, axis=0, weights=weights)
            y_mean = float(np.average(y, weights=weights))
            X_centered = X - x_mean
            y_centered = y - y_mean
        else:
            x_mean = np.zeros(X.shape[1], dtype=float)
            y_mean = 0.0
            X_centered = X.copy()
            y_centered = y.copy()

        if standardize:
            x_scale = np.sqrt(
                np.sum(weights[:, None] * X_centered**2, axis=0) / weight_sum
            )
            valid = x_scale > np.sqrt(np.finfo(float).eps)
            safe_scale = np.ones_like(x_scale)
            safe_scale[valid] = x_scale[valid]
            X_scaled = X_centered / safe_scale
            X_scaled[:, ~valid] = 0.0
        else:
            safe_scale = np.ones(X.shape[1], dtype=float)
            X_scaled = X_centered

        return X_scaled, y_centered, x_mean, safe_scale, y_mean

    def _fit_fixed_alpha(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        alpha: float,
    ) -> _LocalFit:
        """Fit one locally weighted Lasso for a fixed penalty."""
        linear_model = import_optional_dependency(
            "sklearn.linear_model", extra="ml", purpose="GWLasso"
        )
        sklearn_exceptions = import_optional_dependency(
            "sklearn.exceptions", extra="ml", purpose="GWLasso"
        )
        Lasso = linear_model.Lasso
        ConvergenceWarning = sklearn_exceptions.ConvergenceWarning
        positive = np.asarray(weights, dtype=float) > 0.0
        if np.count_nonzero(positive) < 2:
            raise ValueError("A local GWL fit requires at least two positive weights.")
        X_local = np.asarray(X[positive], dtype=float)
        y_local = np.asarray(y[positive], dtype=float)
        w_local = np.asarray(weights[positive], dtype=float)

        X_scaled, y_centered, x_mean, x_scale, y_mean = self._weighted_moments(
            X_local,
            y_local,
            w_local,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
        )

        if alpha <= 0.0:
            sqrt_weights = np.sqrt(w_local)
            design = X_scaled * sqrt_weights[:, None]
            target = y_centered * sqrt_weights
            beta_scaled = np.linalg.lstsq(design, target, rcond=None)[0]
            n_iter = 1
            converged = True
        elif not np.any(np.abs(X_scaled) > 0.0):
            beta_scaled = np.zeros(X.shape[1], dtype=float)
            n_iter = 0
            converged = True
        else:
            model = Lasso(
                alpha=float(alpha),
                fit_intercept=False,
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                selection="cyclic",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_scaled, y_centered, sample_weight=w_local)
            beta_scaled = np.asarray(model.coef_, dtype=float)
            n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
            converged = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )

        coefficients = beta_scaled / x_scale
        intercept = (
            float(y_mean - np.dot(x_mean, coefficients)) if self.fit_intercept else 0.0
        )
        fitted = intercept + X_local @ coefficients
        residual = y_local - fitted
        weight_sum = float(np.sum(w_local))
        objective = float(
            0.5 * np.dot(w_local, residual**2) / weight_sum
            + float(alpha) * np.sum(np.abs(beta_scaled))
        )
        active = np.flatnonzero(np.abs(coefficients) > float(self.active_tol))
        return _LocalFit(
            coefficients=coefficients,
            intercept=intercept,
            alpha=float(alpha),
            objective=objective,
            n_iter=n_iter,
            converged=converged,
            active=active,
            cv_score=np.nan,
        )

    def _alpha_candidates(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Return a descending local regularisation path."""
        if self.alpha_grid is not None:
            return np.unique(np.asarray(self.alpha_grid, dtype=float))[::-1]

        positive = weights > 0.0
        X_local = X[positive]
        y_local = y[positive]
        w_local = weights[positive]
        X_scaled, y_centered, _, _, _ = self._weighted_moments(
            X_local,
            y_local,
            w_local,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
        )
        weight_sum = float(np.sum(w_local))
        alpha_max = float(
            np.max(np.abs(X_scaled.T @ (w_local * y_centered))) / weight_sum
        )
        if not np.isfinite(alpha_max) or alpha_max <= np.finfo(float).eps:
            return np.array([0.0], dtype=float)
        alpha_min = max(alpha_max * float(self.alpha_min_ratio), np.finfo(float).tiny)
        return np.geomspace(alpha_max, alpha_min, int(self.n_alphas))

    def _select_local_alpha(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        *,
        seed_offset: int,
    ) -> Tuple[float, float]:
        """Select a local penalty by weighted K-fold prediction error."""
        if not self._alpha_is_cv:
            return float(self.alpha), np.nan

        model_selection = import_optional_dependency(
            "sklearn.model_selection", extra="ml", purpose="GWLasso cross-validation"
        )
        KFold = model_selection.KFold

        positive_indices = np.flatnonzero(weights > 0.0)
        n_positive = positive_indices.size
        folds = min(int(self.cv_folds), n_positive)
        if folds < 2:
            raise ValueError(
                "Local alpha CV requires at least two weighted observations."
            )

        candidates = self._alpha_candidates(X, y, weights)
        if candidates.size == 1 and candidates[0] == 0.0:
            return 0.0, 0.0

        base_seed = 0 if self.random_state is None else int(self.random_state)
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=(base_seed + int(seed_offset)) % (2**32 - 1),
        )
        squared_error = np.zeros(candidates.size, dtype=float)
        validation_weight = np.zeros(candidates.size, dtype=float)

        local_positions = np.arange(n_positive)
        for train_pos, valid_pos in splitter.split(local_positions):
            train_indices = positive_indices[train_pos]
            valid_indices = positive_indices[valid_pos]
            train_weights = np.zeros_like(weights, dtype=float)
            train_weights[train_indices] = weights[train_indices]
            valid_weights = weights[valid_indices]
            for alpha_index, alpha_value in enumerate(candidates):
                fit = self._fit_fixed_alpha(X, y, train_weights, float(alpha_value))
                prediction = fit.intercept + X[valid_indices] @ fit.coefficients
                error = y[valid_indices] - prediction
                squared_error[alpha_index] += float(np.dot(valid_weights, error**2))
                validation_weight[alpha_index] += float(np.sum(valid_weights))

        scores = np.divide(
            squared_error,
            validation_weight,
            out=np.full_like(squared_error, np.inf),
            where=validation_weight > 0.0,
        )
        best_index = int(np.argmin(scores))
        return float(candidates[best_index]), float(scores[best_index])

    def _fit_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        *,
        seed_offset: int,
    ) -> _LocalFit:
        alpha, cv_score = self._select_local_alpha(
            X,
            y,
            weights,
            seed_offset=seed_offset,
        )
        result = self._fit_fixed_alpha(X, y, weights, alpha)
        return _LocalFit(
            coefficients=result.coefficients,
            intercept=result.intercept,
            alpha=result.alpha,
            objective=result.objective,
            n_iter=result.n_iter,
            converged=result.converged,
            active=result.active,
            cv_score=cv_score,
        )

    def _bandwidth_candidates(
        self, distances: np.ndarray, n_features: int
    ) -> np.ndarray:
        """Construct fixed or adaptive CV candidates."""
        n_samples = distances.shape[0]
        minimum_support = max(int(self.cv_folds) + 1, 4)
        if self.adaptive:
            lower = minimum_support + 1  # includes the focal observation in LOOCV
            upper = n_samples
            if self.bandwidth_range is not None:
                lower = max(lower, int(round(self.bandwidth_range[0])))
                upper = min(upper, int(round(self.bandwidth_range[1])))
            if lower > upper:
                raise ValueError("No valid adaptive GWL bandwidth candidates.")
            values = np.linspace(lower, upper, int(self.n_bandwidths))
            return np.unique(np.rint(values).astype(int))

        nonzero_distances = np.array(distances, dtype=float, copy=True)
        np.fill_diagonal(nonzero_distances, np.inf)
        nonzero = np.sort(nonzero_distances, axis=1)
        order = min(minimum_support, n_samples - 1)
        automatic_lower = float(np.nanmax(nonzero[:, order - 1]))
        automatic_upper = float(np.max(distances))
        if self.bandwidth_range is not None:
            lower = float(self.bandwidth_range[0])
            upper = float(self.bandwidth_range[1])
        else:
            lower = max(automatic_lower, np.nextafter(0.0, 1.0))
            upper = automatic_upper
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("No valid fixed GWL bandwidth candidates.")
        if np.isclose(lower, upper):
            return np.array([lower], dtype=float)
        return np.linspace(lower, upper, int(self.n_bandwidths))

    def _score_bandwidth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        bandwidth: Union[float, int],
    ) -> Tuple[float, int]:
        """Return leave-one-out RMSE and failed local-fit count."""
        predictions = np.full(y.shape[0], np.nan, dtype=float)
        failures = 0
        for location in range(y.shape[0]):
            weights = self._weights(distances[location], bandwidth)
            weights[location] = 0.0
            try:
                local = self._fit_local(
                    X,
                    y,
                    weights,
                    seed_offset=100_000 + location,
                )
                predictions[location] = local.intercept + np.dot(
                    X[location], local.coefficients
                )
            except (ValueError, np.linalg.LinAlgError):
                failures += 1
        if failures or not np.all(np.isfinite(predictions)):
            return np.inf, failures
        return float(np.sqrt(np.mean((y - predictions) ** 2))), 0

    def _select_bandwidth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
    ) -> Union[float, int]:
        """Select bandwidth using global leave-one-out prediction error."""
        candidates = self._bandwidth_candidates(distances, X.shape[1])
        rows = []
        for index, candidate in enumerate(candidates):
            bandwidth_value: Union[float, int]
            bandwidth_value = int(candidate) if self.adaptive else float(candidate)
            score, failures = self._score_bandwidth(
                X,
                y,
                distances,
                bandwidth_value,
            )
            rows.append(
                {
                    "bandwidth": bandwidth_value,
                    "rmse": score,
                    "failed_locations": failures,
                }
            )
            if self.verbose:
                print(
                    f"GWLasso bandwidth {bandwidth_value}: " f"LOOCV RMSE={score:.6g}"
                )
        table = pd.DataFrame(rows)
        self.bandwidth_scores_ = table
        finite = np.isfinite(table["rmse"].to_numpy(dtype=float))
        if not np.any(finite):
            raise RuntimeError(
                "GWLasso bandwidth selection failed for every candidate."
            )
        best_row = (
            table.loc[finite]
            .sort_values(["rmse", "bandwidth"], ascending=[True, False])
            .iloc[0]
        )
        return (
            int(best_row["bandwidth"])
            if self.adaptive
            else float(best_row["bandwidth"])
        )

    def _resolve_bandwidth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
    ) -> Union[float, int]:
        if self.bandwidth is None or isinstance(self.bandwidth, str):
            token = "cv" if self.bandwidth is None else self.bandwidth.strip().lower()
            if token not in {"cv", "adaptive"}:
                raise ValueError(
                    "GWLasso automatic bandwidth must use 'cv' or 'adaptive'."
                )
            return self._select_bandwidth(X, y, distances)

        if self.adaptive:
            value = float(self.bandwidth)
            if not value.is_integer():
                raise ValueError("An adaptive GWL bandwidth must be an integer count.")
            k = int(value)
            if k < 2 or k > X.shape[0]:
                raise ValueError(
                    f"Adaptive bandwidth must be between 2 and {X.shape[0]}."
                )
            return k
        return float(self.bandwidth)

    def _fit_at_locations(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        distance_matrix: np.ndarray,
        *,
        seed_start: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Fit local Lasso models for each row of a distance matrix."""
        n_locations = distance_matrix.shape[0]
        coefficients = np.zeros((n_locations, X_train.shape[1]), dtype=float)
        intercepts = np.zeros(n_locations, dtype=float)
        alphas = np.zeros(n_locations, dtype=float)
        active: List[np.ndarray] = []
        objectives = np.zeros(n_locations, dtype=float)
        iterations = np.zeros(n_locations, dtype=int)
        converged = np.zeros(n_locations, dtype=bool)
        alpha_cv_scores = np.full(n_locations, np.nan, dtype=float)

        if self.bandwidth_ is None:
            raise RuntimeError("bandwidth_ is not set.")
        for location in range(n_locations):
            weights = self._weights(distance_matrix[location], self.bandwidth_)
            local = self._fit_local(
                X_train,
                y_train,
                weights,
                seed_offset=seed_start + location,
            )
            coefficients[location] = local.coefficients
            intercepts[location] = local.intercept
            alphas[location] = local.alpha
            active.append(local.active)
            objectives[location] = local.objective
            iterations[location] = local.n_iter
            converged[location] = local.converged
            alpha_cv_scores[location] = local.cv_score
            if self.verbose and (location + 1) % 50 == 0:
                print(f"GWLasso fitted {location + 1}/{n_locations} locations")

        return (
            coefficients,
            intercepts,
            alphas,
            active,
            objectives,
            iterations,
            converged,
            alpha_cv_scores,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> "GWLasso":
        """Fit local geographically weighted Lasso models."""
        self._reset_gw_lasso_state()
        try:
            X_arr, y_arr, coords_arr = self._validate_inputs(X, y, coords)
            if X_arr.shape[0] < 4:
                raise ValueError("GWLasso requires at least four observations.")
            self._store_training_data(X_arr, y_arr, coords_arr)
            self.kernel_func_ = get_kernel_function(self.kernel)
            distances = compute_distance_matrix(
                coords_arr,
                coords_arr,
                metric=self.distance_metric,
            )
            self.bandwidth_ = self._resolve_bandwidth(
                X_arr,
                y_arr,
                distances,
            )
            (
                self.coef_,
                self.intercept_,
                self.alpha_,
                self.active_vars_,
                self.local_objective_,
                self.n_iter_,
                self.converged_,
                self.local_alpha_cv_score_,
            ) = self._fit_at_locations(
                X_arr,
                y_arr,
                distances,
                seed_start=0,
            )
            self.fitted_values_ = self.intercept_ + np.einsum(
                "ij,ij->i",
                X_arr,
                self.coef_,
            )
            self.residuals_ = y_arr - self.fitted_values_
            selected = np.abs(self.coef_) > float(self.active_tol)
            self.selection_frequency_ = np.mean(selected, axis=0)
            self.mean_active_variables_ = float(np.mean(np.sum(selected, axis=1)))
            approximate_params = self.mean_active_variables_ + (
                1.0 if self.fit_intercept else 0.0
            )
            self.diagnostics_ = compute_diagnostics(
                y_arr,
                self.fitted_values_,
                n_features=approximate_params,
            )
            self.diagnostics_.update(
                {
                    "mean_active_variables": self.mean_active_variables_,
                    "all_local_fits_converged": bool(np.all(self.converged_)),
                    "bandwidth": float(self.bandwidth_),
                    "adaptive": bool(self.adaptive),
                    "mean_alpha": float(np.mean(self.alpha_)),
                }
            )
            self.parameter_names_ = self._feature_names()
            self._mark_fitted()
            return self
        except Exception:
            self._reset_gw_lasso_state()
            raise

    def predict_parameters(
        self,
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> GWLassoPredictionResult:
        """Estimate local coefficients at arbitrary coordinates."""
        self._check_is_fitted()
        if self.X_train_ is None or self.y_train_ is None or self.coords_train_ is None:
            raise RuntimeError("Stored training data are required for prediction.")
        from pygwrx.core.utils import validate_coords

        coords_arr = validate_coords(coords)
        distances = compute_distance_matrix(
            coords_arr,
            self.coords_train_,
            metric=self.distance_metric,
        )
        (
            coefficients,
            intercepts,
            alphas,
            active,
            _,
            _,
            _,
            _,
        ) = self._fit_at_locations(
            self.X_train_,
            self.y_train_,
            distances,
            seed_start=1_000_000,
        )
        return GWLassoPredictionResult(
            predictions=None,
            coefficients=coefficients,
            intercepts=intercepts,
            alphas=alphas,
            active_variables=tuple(active),
            coords=coords_arr,
            feature_names=self._feature_names(),
        )

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Predict by calibrating a weighted Lasso at each new location."""
        self._check_is_fitted()
        X_arr, coords_arr = self._validate_prediction_inputs(X, coords)
        result = self.predict_parameters(coords_arr)
        return result.intercepts + np.einsum(
            "ij,ij->i",
            X_arr,
            result.coefficients,
        )

    def get_variable_importance(self) -> np.ndarray:
        """Return local selection frequency for every predictor."""
        self._check_is_fitted()
        if self.selection_frequency_ is None:
            raise RuntimeError("selection_frequency_ is unavailable.")
        return self.selection_frequency_.copy()

    def to_frame(self) -> pd.DataFrame:
        """Return fitted local coefficients, selections, and residuals."""
        self._check_is_fitted()
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.alpha_ is None
            or self.coords_train_ is None
            or self.fitted_values_ is None
            or self.residuals_ is None
            or self.active_vars_ is None
        ):
            raise RuntimeError("Fitted GWLasso results are incomplete.")
        result = GWLassoPredictionResult(
            predictions=self.fitted_values_,
            coefficients=self.coef_,
            intercepts=self.intercept_,
            alphas=self.alpha_,
            active_variables=tuple(self.active_vars_),
            coords=self.coords_train_,
            feature_names=self._feature_names(),
        ).to_frame()
        result["residual"] = self.residuals_
        result["objective"] = self.local_objective_
        result["converged"] = self.converged_
        return result


__all__ = ["GWLasso", "GWLassoPredictionResult"]
