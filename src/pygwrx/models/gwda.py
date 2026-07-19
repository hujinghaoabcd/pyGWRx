# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Geographically weighted discriminant analysis.

This module implements geographically weighted linear and quadratic
classification using the local means, covariance matrices, and prior
probabilities described by Brunsdon et al. (2007) and the maintained
``GWmodel::gwda`` implementation.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pygwrx.core._summary import format_summary
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import compute_distance_matrix, validate_coords


class GWDA:
    r"""Fit geographically weighted linear or quadratic discriminant analysis.

    Args:
        kernel: Spatial kernel name or callable accepted by
            :func:`pygwrx.core.kernels.get_kernel_function`.
        bandwidth: Positive fixed distance or, when ``adaptive=True``, a
            positive integer neighbour count. ``None`` or ``"cv"`` selects a
            bandwidth by maximizing leave-one-out classification accuracy.
        adaptive: Whether ``bandwidth`` represents a nearest-neighbour count.
        quadratic: Whether to use class-specific covariance matrices (WQDA).
            The default uses a locally pooled covariance matrix (WLDA).
        local_mean: Whether class means vary geographically.
        local_cov: Whether class covariance matrices vary geographically.
        local_prior: Whether class prior probabilities vary geographically.
        prior: Optional fixed class priors in sorted class-label order. Values
            must be non-negative and sum to one.
        regularization: Explicit non-negative ridge added to covariance
            diagonals. The default performs the published unregularized method
            and raises when a required covariance is singular.
        verbose: Whether to print a compact completion message.

    Notes:
        For class :math:`g` at prediction location :math:`u`, pyGWRx computes
        a local weighted mean :math:`\mu_g(u)`, an unbiased weighted covariance
        :math:`\Sigma_g(u)`, and a local prior :math:`\pi_g(u)`. The Gaussian
        discriminant cost is

        .. math::

            d_g(x,u) = \tfrac12\log|\Sigma_g(u)|
            + \tfrac12(x-\mu_g(u))^T\Sigma_g(u)^{-1}(x-\mu_g(u))
            - \log\pi_g(u).

        WLDA replaces the class-specific covariance by a locally pooled
        covariance. Classification selects the class with minimum cost.

        ``GWmodel::gwda`` uses the same local statistics and classification
        ordering, but its published R source multiplies a matrix-norm term by
        the number of classes. pyGWRx uses the standard Gaussian log-determinant
        formula so that returned probabilities have a clear statistical meaning.
    """

    _FIT_ATTRIBUTES = (
        "classes_",
        "feature_names_in_",
        "n_features_in_",
        "class_counts_",
        "fixed_prior_",
        "bandwidth_",
        "bandwidth_scores_",
        "X_train_",
        "y_train_",
        "coords_train_",
        "class_means_",
        "class_covariances_",
        "class_covs_",
        "class_priors_",
        "pooled_covariances_",
        "predictions_",
        "probabilities_",
        "discriminant_scores_",
        "entropy_",
        "confusion_matrix_",
        "correct_ratio_",
        "validation_mode_",
    )

    def __init__(
        self,
        kernel: str | Any = "bisquare",
        bandwidth: float | int | str | None = "cv",
        adaptive: bool = True,
        quadratic: bool = False,
        local_mean: bool = True,
        local_cov: bool = True,
        local_prior: bool = True,
        prior: np.ndarray | list[float] | tuple[float, ...] | None = None,
        regularization: float = 0.0,
        verbose: bool = False,
    ) -> None:
        for name, value in {
            "adaptive": adaptive,
            "quadratic": quadratic,
            "local_mean": local_mean,
            "local_cov": local_cov,
            "local_prior": local_prior,
            "verbose": verbose,
        }.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
        if isinstance(regularization, (bool, np.bool_)) or not isinstance(
            regularization, Real
        ):
            raise TypeError("regularization must be a non-negative real number.")
        if not np.isfinite(float(regularization)) or float(regularization) < 0:
            raise ValueError("regularization must be finite and non-negative.")

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.adaptive = bool(adaptive)
        self.quadratic = bool(quadratic)
        self.local_mean = bool(local_mean)
        self.local_cov = bool(local_cov)
        self.local_prior = bool(local_prior)
        self.prior = prior
        self.regularization = float(regularization)
        self.verbose = bool(verbose)
        self._is_fitted = False
        self._clear_fit_state()

    def _clear_fit_state(self) -> None:
        """Remove every result created by a previous fit attempt."""
        for attribute in self._FIT_ATTRIBUTES:
            setattr(self, attribute, None)
        self._is_fitted = False

    @staticmethod
    def _validate_X(
        X: np.ndarray | pd.DataFrame,
        *,
        expected_features: int | None = None,
        name: str = "X",
    ) -> tuple[np.ndarray, list[str] | None]:
        """Validate a finite two-dimensional numeric feature matrix."""
        columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        try:
            array = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain only numeric values.") from exc
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")
        if array.shape[0] < 1 or array.shape[1] < 2:
            raise ValueError(f"{name} must contain at least one row and two variables.")
        if expected_features is not None and array.shape[1] != expected_features:
            raise ValueError(
                f"{name} has {array.shape[1]} variables; expected {expected_features}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array, columns

    @staticmethod
    def _validate_y(y: np.ndarray | pd.Series, n_samples: int) -> np.ndarray:
        """Validate a one-dimensional class-label vector without missing values."""
        array = np.asarray(y)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if array.shape[0] != n_samples:
            raise ValueError("X and y must contain the same number of rows.")
        if pd.isna(array).any():
            raise ValueError("y must not contain missing labels.")
        return array

    def _validate_bandwidth(
        self, bandwidth: float | int, n_samples: int
    ) -> float | int:
        """Validate fixed-distance or adaptive-neighbour bandwidth semantics."""
        if self.adaptive:
            if isinstance(bandwidth, (bool, np.bool_)) or not isinstance(
                bandwidth, Integral
            ):
                raise TypeError(
                    "An adaptive bandwidth must be an integer neighbour count."
                )
            value = int(bandwidth)
            if value < 2 or value > n_samples:
                raise ValueError(
                    f"An adaptive bandwidth must be between 2 and {n_samples}."
                )
            return value
        if isinstance(bandwidth, (bool, np.bool_)) or not isinstance(bandwidth, Real):
            raise TypeError("A fixed bandwidth must be a positive real distance.")
        value = float(bandwidth)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("A fixed bandwidth must be positive and finite.")
        return value

    def _weights(self, distances: np.ndarray, bandwidth: float | int) -> np.ndarray:
        """Construct fixed or GWmodel-compatible adaptive kernel weights."""
        kernel = get_kernel_function(self.kernel)
        distances_arr = np.asarray(distances, dtype=float)
        if not self.adaptive:
            return np.asarray(kernel(distances_arr, float(bandwidth)), dtype=float)

        k = int(bandwidth)
        order = np.argsort(distances_arr, kind="stable")
        kernel_name = (
            self.kernel.strip().lower() if isinstance(self.kernel, str) else None
        )
        if kernel_name == "boxcar":
            weights = np.zeros_like(distances_arr, dtype=float)
            weights[order[:k]] = 1.0
            return weights
        kth_distance = float(distances_arr[order[k - 1]])
        if kth_distance == 0.0:
            weights = np.zeros_like(distances_arr, dtype=float)
            weights[order[:k]] = 1.0
            return weights
        return np.asarray(kernel(distances_arr, kth_distance), dtype=float)

    def _validate_prior(self, classes: np.ndarray) -> np.ndarray | None:
        """Validate fixed prior probabilities in sorted class order."""
        if self.prior is None:
            return None
        prior = np.asarray(self.prior, dtype=float)
        if prior.ndim != 1 or prior.shape[0] != classes.shape[0]:
            raise ValueError(
                "prior must contain one value for each sorted class label."
            )
        if not np.all(np.isfinite(prior)) or np.any(prior < 0):
            raise ValueError("prior values must be finite and non-negative.")
        if not np.isclose(np.sum(prior), 1.0, atol=1e-10):
            raise ValueError("prior values must sum to one.")
        if np.any(prior == 0):
            raise ValueError("Every formally supported class prior must be positive.")
        return prior

    @staticmethod
    def _weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute a normalized weighted mean and reject empty local support."""
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("A class has zero local kernel weight.")
        return np.sum(X * weights[:, None], axis=0) / total

    @staticmethod
    def _weighted_covariance(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute the unbiased normalized covariance used by ``stats::cov.wt``."""
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("A class has zero local kernel weight.")
        normalized = np.asarray(weights, dtype=float) / total
        denominator = 1.0 - float(np.sum(normalized**2))
        if denominator <= np.finfo(float).eps:
            raise ValueError(
                "Local class support is insufficient for covariance estimation."
            )
        mean = np.sum(X * normalized[:, None], axis=0)
        centered = X - mean
        return (centered.T * normalized) @ centered / denominator

    def _regularize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Apply only the explicitly requested diagonal covariance ridge."""
        covariance = np.asarray(covariance, dtype=float)
        covariance = (covariance + covariance.T) / 2.0
        if self.regularization > 0:
            covariance = covariance + self.regularization * np.eye(covariance.shape[0])
        return covariance

    @staticmethod
    def _factor_covariance(covariance: np.ndarray) -> tuple[np.ndarray, float]:
        """Return an inverse and log determinant for a positive-definite covariance."""
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(log_determinant):
            raise np.linalg.LinAlgError(
                "A required covariance matrix is singular or not positive definite. "
                "Increase the bandwidth or set an explicit positive regularization."
            )
        try:
            inverse = np.linalg.inv(covariance)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "A required covariance matrix could not be inverted."
            ) from exc
        return inverse, float(log_determinant)

    def _statistics_at_locations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_eval: np.ndarray,
        weights: np.ndarray,
        classes: np.ndarray,
        fixed_prior: np.ndarray | None,
    ) -> dict[str, Any]:
        """Compute local statistics, Gaussian costs, labels, and probabilities."""
        n_eval = X_eval.shape[0]
        n_features = X.shape[1]
        n_classes = classes.shape[0]
        class_indices = [np.flatnonzero(y == label) for label in classes]
        class_counts = np.asarray(
            [indices.size for indices in class_indices], dtype=int
        )

        means: dict[Any, np.ndarray] = {
            label: np.empty((n_eval, n_features), dtype=float) for label in classes
        }
        covariances: dict[Any, np.ndarray] = {
            label: np.empty((n_eval, n_features, n_features), dtype=float)
            for label in classes
        }
        priors: dict[Any, np.ndarray] = {
            label: np.empty(n_eval, dtype=float) for label in classes
        }
        pooled = (
            None
            if self.quadratic
            else np.empty((n_eval, n_features, n_features), dtype=float)
        )
        costs = np.empty((n_eval, n_classes), dtype=float)

        global_weights = np.ones(X.shape[0], dtype=float)
        for location in range(n_eval):
            local_weights = weights[:, location]
            total_local_weight = float(np.sum(local_weights))
            if not np.isfinite(total_local_weight) or total_local_weight <= 0:
                raise ValueError(
                    f"Evaluation location {location} has zero kernel weight."
                )

            location_covariances: list[np.ndarray] = []
            location_means: list[np.ndarray] = []
            location_priors: list[float] = []
            for class_position, (label, indices) in enumerate(
                zip(classes, class_indices, strict=True)
            ):
                class_X = X[indices]
                class_local_weights = local_weights[indices]
                mean_weights = (
                    class_local_weights if self.local_mean else global_weights[indices]
                )
                cov_weights = (
                    class_local_weights if self.local_cov else global_weights[indices]
                )
                mean = self._weighted_mean(class_X, mean_weights)
                covariance = self._weighted_covariance(class_X, cov_weights)

                if fixed_prior is not None:
                    prior = float(fixed_prior[class_position])
                elif self.local_prior:
                    prior = float(np.sum(class_local_weights) / total_local_weight)
                else:
                    prior = float(indices.size / X.shape[0])
                if not np.isfinite(prior) or prior <= 0:
                    raise ValueError(
                        f"Class {label!r} has a non-positive prior at evaluation "
                        f"location {location}; increase the bandwidth."
                    )

                means[label][location] = mean
                covariances[label][location] = covariance
                priors[label][location] = prior
                location_means.append(mean)
                location_covariances.append(covariance)
                location_priors.append(prior)

            if self.quadratic:
                location_covariances = [
                    self._regularize_covariance(covariance)
                    for covariance in location_covariances
                ]
                for class_position, label in enumerate(classes):
                    covariances[label][location] = location_covariances[class_position]
                covariance_factors = [
                    self._factor_covariance(covariance)
                    for covariance in location_covariances
                ]
            else:
                # GWmodel pools class covariance matrices with the global class counts.
                pooled_covariance = np.zeros((n_features, n_features), dtype=float)
                for count, covariance in zip(
                    class_counts, location_covariances, strict=True
                ):
                    pooled_covariance += count * covariance
                pooled_covariance /= float(np.sum(class_counts))
                pooled_covariance = self._regularize_covariance(pooled_covariance)
                pooled[location] = pooled_covariance
                common_factor = self._factor_covariance(pooled_covariance)
                covariance_factors = [common_factor] * n_classes

            x = X_eval[location]
            for class_position, (mean, prior, factor) in enumerate(
                zip(location_means, location_priors, covariance_factors, strict=True)
            ):
                inverse, log_determinant = factor
                difference = x - mean
                mahalanobis = float(difference @ inverse @ difference)
                costs[location, class_position] = (
                    0.5 * log_determinant + 0.5 * mahalanobis - np.log(prior)
                )

        predictions = classes[np.argmin(costs, axis=1)]
        shifted = -costs - np.max(-costs, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
        if n_classes == 1:
            entropy = np.zeros(n_eval, dtype=float)
        else:
            safe = np.clip(probabilities, np.finfo(float).tiny, 1.0)
            entropy = -np.sum(safe * np.log2(safe), axis=1) / np.log2(n_classes)

        return {
            "means": means,
            "covariances": covariances,
            "priors": priors,
            "pooled": pooled,
            "costs": costs,
            "predictions": predictions,
            "probabilities": probabilities,
            "entropy": entropy,
        }

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        X_eval: np.ndarray,
        coords_eval: np.ndarray,
        bandwidth: float | int,
        *,
        leave_one_out: bool,
        classes: np.ndarray,
        fixed_prior: np.ndarray | None,
    ) -> dict[str, Any]:
        """Evaluate GWDA at supplied locations without mutating estimator state."""
        distances = compute_distance_matrix(coords_eval, coords)
        weights = np.column_stack(
            [
                self._weights(distances[index], bandwidth)
                for index in range(X_eval.shape[0])
            ]
        )
        if leave_one_out:
            if X_eval.shape[0] != X.shape[0] or not np.array_equal(coords_eval, coords):
                raise ValueError(
                    "leave_one_out requires the training rows and locations."
                )
            np.fill_diagonal(weights, 0.0)
        return self._statistics_at_locations(
            X, y, X_eval, weights, classes, fixed_prior
        )

    def _cv_accuracy(
        self,
        bandwidth: float | int,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        classes: np.ndarray,
        fixed_prior: np.ndarray | None,
    ) -> float:
        """Return leave-one-out classification accuracy or zero on local failure."""
        try:
            result = self._evaluate(
                X,
                y,
                coords,
                X,
                coords,
                bandwidth,
                leave_one_out=True,
                classes=classes,
                fixed_prior=fixed_prior,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 0.0
        return float(np.mean(result["predictions"] == y))

    def select_bandwidth(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        coords: np.ndarray | pd.DataFrame,
        *,
        bounds: tuple[float | int, float | int] | None = None,
    ) -> float | int:
        """Select a bandwidth by maximizing leave-one-out accuracy.

        Args:
            X: Training feature matrix.
            y: Training class labels.
            coords: Training coordinates.
            bounds: Optional closed search interval. Adaptive bounds are integer
                neighbour counts; fixed bounds are positive distances.

        Returns:
            Selected bandwidth.
        """
        X_array, _ = self._validate_X(X)
        y_array = self._validate_y(y, X_array.shape[0])
        coords_array = validate_coords(coords)
        if coords_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        classes = np.unique(y_array)
        if classes.size < 2:
            raise ValueError("GWDA requires at least two classes.")
        fixed_prior = self._validate_prior(classes)
        self._validate_class_sizes(y_array, classes, X_array.shape[1])

        n_samples = X_array.shape[0]
        distance_matrix = compute_distance_matrix(coords_array, coords_array)
        if bounds is None:
            if self.adaptive:
                lower = max(2, min(n_samples, max(20, X_array.shape[1] + 2)))
                upper = n_samples
            else:
                maximum = float(np.max(distance_matrix))
                if maximum <= 0:
                    raise ValueError(
                        "Coordinates must contain at least two distinct locations."
                    )
                lower = maximum / 5000.0
                upper = maximum
        else:
            if len(bounds) != 2:
                raise ValueError("bounds must contain exactly two values.")
            lower = self._validate_bandwidth(bounds[0], n_samples)
            upper = self._validate_bandwidth(bounds[1], n_samples)
            if lower >= upper:
                raise ValueError(
                    "The lower bandwidth bound must be smaller than the upper bound."
                )

        cache: dict[float | int, float] = {}

        def score(candidate: float | int) -> float:
            bandwidth = int(round(candidate)) if self.adaptive else float(candidate)
            bandwidth = self._validate_bandwidth(bandwidth, n_samples)
            if bandwidth not in cache:
                cache[bandwidth] = self._cv_accuracy(
                    bandwidth,
                    X_array,
                    y_array,
                    coords_array,
                    classes,
                    fixed_prior,
                )
            return cache[bandwidth]

        if self.adaptive and int(upper) - int(lower) <= 180:
            for candidate in range(int(lower), int(upper) + 1):
                score(candidate)
        else:
            optimization = minimize_scalar(
                lambda value: -score(value),
                bounds=(float(lower), float(upper)),
                method="bounded",
                options={"xatol": 1.0 if self.adaptive else 1e-5},
            )
            center = (
                int(round(optimization.x)) if self.adaptive else float(optimization.x)
            )
            if self.adaptive:
                for candidate in range(
                    max(int(lower), center - 4), min(int(upper), center + 4) + 1
                ):
                    score(candidate)
            else:
                score(center)

        best = min(cache, key=lambda candidate: (-cache[candidate], candidate))
        self.bandwidth_scores_ = tuple(sorted(cache.items(), key=lambda item: item[0]))
        return best

    @staticmethod
    def _validate_class_sizes(
        y: np.ndarray, classes: np.ndarray, n_features: int
    ) -> None:
        """Require enough observations for full covariance estimation."""
        counts = np.asarray([np.sum(y == label) for label in classes], dtype=int)
        minimum = n_features + 1
        if np.any(counts < minimum):
            failing = [
                repr(label) for label, count in zip(classes, counts) if count < minimum
            ]
            raise ValueError(
                "Each class must contain at least n_features + 1 observations; "
                f"insufficient classes: {', '.join(failing)}."
            )

    @staticmethod
    def _confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
    ) -> np.ndarray:
        """Return GWmodel-style predicted-row/observed-column counts and totals."""
        matrix = np.zeros((classes.size + 1, classes.size + 1), dtype=int)
        for row, predicted in enumerate(classes):
            for column, observed in enumerate(classes):
                matrix[row, column] = int(
                    np.sum((y_pred == predicted) & (y_true == observed))
                )
        matrix[:-1, -1] = np.sum(matrix[:-1, :-1], axis=1)
        matrix[-1, :-1] = np.sum(matrix[:-1, :-1], axis=0)
        matrix[-1, -1] = int(y_true.size)
        return matrix

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        coords: np.ndarray | pd.DataFrame,
        X_pred: np.ndarray | pd.DataFrame | None = None,
        coords_pred: np.ndarray | pd.DataFrame | None = None,
        validate: bool = True,
    ) -> "GWDA":
        """Fit GWDA and optionally evaluate training or supplied prediction rows.

        When prediction rows are omitted, ``validate=True`` performs leave-one-out
        classification and stores accuracy and a GWmodel-style confusion matrix.
        Supplying ``X_pred`` and ``coords_pred`` performs ordinary prediction while
        retaining the fitted training data for later calls to :meth:`predict`.
        """
        self._clear_fit_state()
        try:
            X_array, columns = self._validate_X(X)
            y_array = self._validate_y(y, X_array.shape[0])
            coords_array = validate_coords(coords)
            if coords_array.shape[0] != X_array.shape[0]:
                raise ValueError("X and coords must contain the same number of rows.")
            classes = np.unique(y_array)
            if classes.size < 2:
                raise ValueError("GWDA requires at least two classes.")
            self._validate_class_sizes(y_array, classes, X_array.shape[1])
            fixed_prior = self._validate_prior(classes)

            if self.bandwidth is None or (
                isinstance(self.bandwidth, str)
                and self.bandwidth.strip().lower() == "cv"
            ):
                bandwidth = self.select_bandwidth(X_array, y_array, coords_array)
            elif isinstance(self.bandwidth, str):
                raise ValueError("bandwidth must be numeric, None, or 'cv'.")
            else:
                bandwidth = self._validate_bandwidth(self.bandwidth, X_array.shape[0])

            if X_pred is None and coords_pred is None:
                X_eval = X_array
                coords_eval = coords_array
                leave_one_out = bool(validate)
                validation_mode = "leave-one-out" if validate else "training"
            elif X_pred is None or coords_pred is None:
                raise ValueError("X_pred and coords_pred must be supplied together.")
            else:
                X_eval, prediction_columns = self._validate_X(
                    X_pred,
                    expected_features=X_array.shape[1],
                    name="X_pred",
                )
                if (
                    columns is not None
                    and prediction_columns is not None
                    and columns != prediction_columns
                ):
                    raise ValueError(
                        "X_pred DataFrame columns must match the training columns."
                    )
                coords_eval = validate_coords(coords_pred)
                if coords_eval.shape[0] != X_eval.shape[0]:
                    raise ValueError(
                        "X_pred and coords_pred must contain the same number of rows."
                    )
                leave_one_out = False
                validation_mode = "prediction"

            result = self._evaluate(
                X_array,
                y_array,
                coords_array,
                X_eval,
                coords_eval,
                bandwidth,
                leave_one_out=leave_one_out,
                classes=classes,
                fixed_prior=fixed_prior,
            )

            self.classes_ = classes
            self.feature_names_in_ = columns
            self.n_features_in_ = X_array.shape[1]
            self.fixed_prior_ = None if fixed_prior is None else fixed_prior.copy()
            self.class_counts_ = np.asarray(
                [np.sum(y_array == label) for label in classes], dtype=int
            )
            self.bandwidth_ = bandwidth
            self.X_train_ = X_array.copy()
            self.y_train_ = y_array.copy()
            self.coords_train_ = coords_array.copy()
            self.class_means_ = result["means"]
            self.class_covariances_ = result["covariances"]
            # Backward-compatible spelling retained as a documented alias.
            self.class_covs_ = self.class_covariances_
            self.class_priors_ = result["priors"]
            self.pooled_covariances_ = result["pooled"]
            self.discriminant_scores_ = result["costs"]
            self.log_posteriors_ = self.discriminant_scores_
            self.predictions_ = result["predictions"]
            self.probabilities_ = result["probabilities"]
            self.entropy_ = result["entropy"]
            self.validation_mode_ = validation_mode
            if validate and X_pred is None:
                self.confusion_matrix_ = self._confusion_matrix(
                    y_array, self.predictions_, classes
                )
                self.correct_ratio_ = float(np.mean(self.predictions_ == y_array))
            self._is_fitted = True
        except Exception:
            self._clear_fit_state()
            raise

        if self.verbose:
            print(
                "GWDA fit complete: "
                f"method={'WQDA' if self.quadratic else 'WLDA'}, "
                f"bandwidth={self.bandwidth_}, mode={self.validation_mode_}."
            )
        return self

    def _check_fitted(self) -> None:
        """Raise a stable error when prediction is requested before fitting."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

    def _predict_result(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluate new locations without modifying fitted validation results."""
        self._check_fitted()
        X_array, columns = self._validate_X(
            X, expected_features=self.n_features_in_, name="X"
        )
        if (
            self.feature_names_in_ is not None
            and columns is not None
            and columns != self.feature_names_in_
        ):
            raise ValueError(
                "Prediction DataFrame columns must match the training columns."
            )
        coords_array = validate_coords(coords)
        if coords_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        fixed_prior = self.fixed_prior_
        return self._evaluate(
            self.X_train_,
            self.y_train_,
            self.coords_train_,
            X_array,
            coords_array,
            self.bandwidth_,
            leave_one_out=False,
            classes=self.classes_,
            fixed_prior=fixed_prior,
        )

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Predict class labels at new spatial locations without refitting."""
        return self._predict_result(X, coords)["predictions"]

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Return normalized Gaussian class probabilities at new locations."""
        return self._predict_result(X, coords)["probabilities"]

    def predict_entropy(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Return normalized Shannon classification entropy at new locations."""
        return self._predict_result(X, coords)["entropy"]

    def get_entropy(self) -> np.ndarray:
        """Return entropy stored by the most recent successful fit."""
        self._check_fitted()
        return self.entropy_.copy()

    def summary(self) -> str:
        """Return a plain-text fitted-model summary."""
        self._check_fitted()
        return format_summary(
            "GWDA Summary",
            {
                "n_samples": int(self.X_train_.shape[0]),
                "n_features": int(self.n_features_in_),
                "n_classes": int(self.classes_.size),
                "classes": self.classes_.copy(),
                "method": "WQDA" if self.quadratic else "WLDA",
                "bandwidth": self.bandwidth_,
                "adaptive": self.adaptive,
                "validation_mode": self.validation_mode_,
                "correct_ratio": self.correct_ratio_,
                "mean_entropy": float(np.mean(self.entropy_)),
            },
        )
