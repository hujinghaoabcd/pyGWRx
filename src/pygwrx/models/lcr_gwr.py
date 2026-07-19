# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Locally compensated ridge geographically weighted regression.

This module implements the classical GWR-LCR algorithm exposed by
``GWmodel::gwr.lcr`` while retaining the standard pyGWRx estimator,
diagnostic, inference, prediction, and result-export interfaces.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.optimization import BrentSearch, GoldenSectionSearch
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import add_intercept, compute_distance_matrix, validate_coords
from pygwrx.models.gwr import GWR, GWRPredictionResult


@dataclass
class _LCRLocalFitResult:
    """Internal results from fitting all calibration locations."""

    params: np.ndarray
    fitted_values: np.ndarray
    distances: np.ndarray
    influence: np.ndarray
    trace_S: float
    trace_StS: float
    covariance_factors: Optional[np.ndarray]
    hat_matrix: Optional[np.ndarray]
    condition_numbers: np.ndarray
    compensated_condition_numbers: np.ndarray
    penalized_system_condition_numbers: np.ndarray
    local_lambdas: np.ndarray


class LCRGWR(GWR):
    r"""Locally compensated ridge geographically weighted regression.

    LCR-GWR diagnoses local collinearity from the weighted design matrix and
    applies a location-specific ridge parameter only where the local condition
    number exceeds a user-defined threshold. The compensation rule follows the
    implementation in ``GWmodel::gwr.lcr``::

        lambda_i = (d_max - kappa_star * d_min) / (kappa_star - 1),

    where ``d_max`` and ``d_min`` are the largest and smallest singular values
    of the column-normalized, locally weighted design matrix, and
    ``kappa_star`` is ``cn_thresh``.

    The local coefficient estimator uses the GWmodel scaling convention. With
    ``A = diag(1 / x_scale)`` and local spatial weights ``W_i``, the estimator is

    .. math::

        \hat\beta_i = A\left(A X^T W_i X A + \lambda_i I\right)^{-1}
        A X^T W_i y.

    Unlike the historical GWmodel diagnostics, pyGWRx constructs the hat matrix
    from the actual penalized estimator. Consequently, trace statistics,
    effective degrees of freedom, information criteria, influence, and standard
    errors remain internally consistent when a ridge term is active.

    Args:
        kernel: Spatial kernel name or callable.
        bandwidth: Fixed distance, adaptive neighbour count, ``"cv"``, or
            ``None``. Automatic LCR-GWR bandwidth selection is based on strict
            leave-one-out cross-validation, matching ``bw.gwr.lcr``.
        bandwidth_method: Automatic selection criterion. Only ``"cv"`` is
            supported for the classical LCR-GWR algorithm.
        adaptive: Interpret the bandwidth as an integer neighbour count.
        bandwidth_range: Optional lower and upper bandwidth-search bounds.
        optimization_method: ``"golden_section"``, ``"brent"``, or ``"grid"``.
        lambda_ridge: Constant ridge parameter used at every location before
            optional local compensation. The GWmodel default is ``0``.
        lambda_adjust: Whether to replace ``lambda_ridge`` at locations whose
            local condition number exceeds ``cn_thresh``.
        cn_thresh: Maximum desired local condition number. Values from 20 to 30
            are common in the LCR-GWR literature.
        fit_intercept: Whether to include a local intercept.
        distance_metric: Distance metric used to construct spatial weights.
        sigma2_v1: Residual-variance convention inherited from :class:`GWR`.
        verbose: Whether to print fit and bandwidth-selection progress.

    Attributes:
        condition_numbers_: Local pre-compensation condition numbers using the
            GWmodel/Belsley column-normalization convention.
        local_lambda_: Ridge parameter used at each calibration location.
        compensated_condition_numbers_: Condition numbers implied by the
            GWmodel compensation formula.
        penalized_system_condition_numbers_: Numerical condition numbers of the
            actual penalized normal systems used for estimation.
        locally_compensated_mask_: Boolean mask identifying locations where the
            threshold-triggered local compensation was applied.
        ridge_applied_mask_: Boolean mask identifying all locations with a
            positive ridge parameter.
        cv_residuals_: Leave-one-out residuals when ``compute_cv=True``.
        cv_contributions_: Squared leave-one-out residuals.
        bandwidth_cv_score_: Sum of squared leave-one-out residuals for the
            selected or supplied bandwidth when available.

    Notes:
        The reference GWmodel routine penalizes the intercept together with the
        slopes. This implementation preserves that convention for numerical
        comparability.

        ``condition_numbers_`` are diagnostics of the unpenalized local design;
        they are therefore expected to remain above ``cn_thresh`` at affected
        locations. Use ``compensated_condition_numbers_`` or
        ``penalized_system_condition_numbers_`` to inspect post-penalty systems.

    References:
        Wheeler, D. C. (2007). Diagnostic tools and a remedial method for
        collinearity in geographically weighted regression. *Environment and
        Planning A*, 39(10), 2464-2481.

        Gollini, I., Lu, B., Charlton, M., Brunsdon, C., and Harris, P. (2015).
        GWmodel: An R package for exploring spatial heterogeneity using
        geographically weighted models. *Journal of Statistical Software*,
        63(17), 1-50.
    """

    def __init__(
        self,
        kernel: Union[str, Callable[[np.ndarray, float], np.ndarray]] = "bisquare",
        bandwidth: Union[float, int, str, None] = "cv",
        bandwidth_method: str = "cv",
        adaptive: bool = False,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        optimization_method: str = "golden_section",
        lambda_ridge: float = 0.0,
        lambda_adjust: bool = True,
        cn_thresh: float = 30.0,
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method=bandwidth_method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            sigma2_v1=sigma2_v1,
            verbose=verbose,
        )
        self.lambda_ridge = lambda_ridge
        self.lambda_adjust = lambda_adjust
        self.cn_thresh = cn_thresh
        self._validate_lcr_parameters()
        self._reset_lcr_state()

    def _validate_lcr_parameters(self) -> None:
        if isinstance(self.lambda_ridge, (bool, np.bool_)) or not isinstance(
            self.lambda_ridge, (int, float, np.integer, np.floating)
        ):
            raise TypeError("lambda_ridge must be a finite non-negative number.")
        lambda_value = float(self.lambda_ridge)
        if not np.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError("lambda_ridge must be finite and non-negative.")
        self.lambda_ridge = lambda_value

        if not isinstance(self.lambda_adjust, (bool, np.bool_)):
            raise TypeError("lambda_adjust must be boolean.")
        self.lambda_adjust = bool(self.lambda_adjust)

        if isinstance(self.cn_thresh, (bool, np.bool_)) or not isinstance(
            self.cn_thresh, (int, float, np.integer, np.floating)
        ):
            raise TypeError("cn_thresh must be a finite number greater than one.")
        threshold = float(self.cn_thresh)
        if not np.isfinite(threshold) or threshold <= 1.0:
            raise ValueError("cn_thresh must be finite and greater than one.")
        self.cn_thresh = threshold

    def _reset_lcr_state(self) -> None:
        self.condition_numbers_: Optional[np.ndarray] = None
        self.local_condition_numbers_: Optional[np.ndarray] = None
        self.compensated_condition_numbers_: Optional[np.ndarray] = None
        self.penalized_system_condition_numbers_: Optional[np.ndarray] = None
        self.local_lambda_: Optional[np.ndarray] = None
        self.local_lambdas_: Optional[np.ndarray] = None
        self.locally_compensated_mask_: Optional[np.ndarray] = None
        self.ridge_applied_mask_: Optional[np.ndarray] = None
        self.cv_residuals_: Optional[np.ndarray] = None
        self.cv_contributions_: Optional[np.ndarray] = None
        self.bandwidth_cv_score_: Optional[float] = None
        self.bandwidth_selection_result_: Optional[Dict[str, object]] = None
        self.design_scales_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None

    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self._reset_lcr_state()

    def _compute_design_scales(self, X_design: np.ndarray) -> np.ndarray:
        if X_design.shape[0] < 2:
            raise ValueError("LCRGWR requires at least two observations.")
        if self.fit_intercept:
            scales = np.ones(X_design.shape[1], dtype=float)
            if X_design.shape[1] > 1:
                scales[1:] = np.std(X_design[:, 1:], axis=0, ddof=1)
        else:
            scales = np.std(X_design, axis=0, ddof=1)

        invalid = ~np.isfinite(scales) | (scales <= np.finfo(float).eps)
        if np.any(invalid):
            positions = ", ".join(str(index) for index in np.flatnonzero(invalid))
            raise ValueError(
                "LCRGWR cannot scale constant or non-finite design columns; "
                f"invalid design-column indices: {positions}."
            )
        return scales

    def _weights_for_candidate(
        self,
        distances: np.ndarray,
        bandwidth: Union[int, float],
    ) -> np.ndarray:
        local_bandwidth = (
            adaptive_bandwidth_weights(distances, int(bandwidth))
            if self.adaptive
            else float(bandwidth)
        )
        if self.kernel_func_ is None:
            raise RuntimeError("The kernel function is unavailable.")
        weights = np.asarray(self.kernel_func_(distances, local_bandwidth), dtype=float)
        if weights.shape != distances.shape:
            raise ValueError("The kernel returned an unexpected weight shape.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("The kernel returned invalid weights.")
        return weights

    def _condition_and_lambda(
        self,
        X_design: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray]:
        # GWmodel uses W X (rather than sqrt(W) X) for this Belsley-style
        # condition-number diagnostic. Preserve that convention exactly.
        weighted_design = weights[:, None] * X_design
        column_norms = np.sqrt(np.sum(weighted_design**2, axis=0))
        eps = np.finfo(float).eps
        normalized = np.zeros_like(weighted_design, dtype=float)
        np.divide(
            weighted_design,
            column_norms,
            out=normalized,
            where=column_norms > eps,
        )
        # A locally constant or unsupported predictor produces a zero singular
        # value. Keeping that zero in the normalized design yields an infinite
        # pre-compensation condition number and lets the classical LCR formula
        # choose a positive stabilizing ridge parameter.
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        condition = np.inf if smallest <= eps else largest / smallest

        local_lambda = float(self.lambda_ridge)
        if self.lambda_adjust and condition > self.cn_thresh:
            local_lambda = max(
                0.0,
                (largest - self.cn_thresh * smallest) / (self.cn_thresh - 1.0),
            )

        denominator = smallest + local_lambda
        compensated = (
            np.inf if denominator <= eps else (largest + local_lambda) / denominator
        )
        return condition, local_lambda, compensated, singular_values

    @staticmethod
    def _ridge_transform(
        X_design: np.ndarray,
        weights: np.ndarray,
        design_scales: np.ndarray,
        local_lambda: float,
    ) -> Tuple[np.ndarray, float]:
        scaled_design = X_design / design_scales
        weighted_normal = scaled_design.T @ (weights[:, None] * scaled_design)
        penalized_normal = weighted_normal + local_lambda * np.eye(
            X_design.shape[1], dtype=float
        )
        inverse = np.linalg.pinv(penalized_normal)
        transform_scaled = inverse @ (scaled_design.T * weights)
        transform = transform_scaled / design_scales[:, None]
        system_condition = float(np.linalg.cond(penalized_normal))
        return transform, system_condition

    def _fit_one_location(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        target_row: np.ndarray,
        design_scales: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        if not np.any(weights > 0.0):
            raise ValueError("The local kernel contains no positive weights.")
        condition, local_lambda, compensated, _ = self._condition_and_lambda(
            X_design, weights
        )
        transform, system_condition = self._ridge_transform(
            X_design,
            weights,
            design_scales,
            local_lambda,
        )
        beta = transform @ y
        hat_row = target_row @ transform
        return (
            beta,
            hat_row,
            condition,
            local_lambda,
            compensated,
            system_condition,
        )

    def _fit_training_locations_lcr(
        self,
        X_design: np.ndarray,
        distances: np.ndarray,
        *,
        store_hat_matrix: bool,
        compute_inference: bool,
    ) -> _LCRLocalFitResult:
        if self.y_train_ is None or self.design_scales_ is None:
            raise RuntimeError("Training data and scaling state are unavailable.")

        n_samples, n_parameters = X_design.shape
        params = np.empty((n_samples, n_parameters), dtype=float)
        fitted = np.empty(n_samples, dtype=float)
        influence = np.empty(n_samples, dtype=float)
        condition_numbers = np.empty(n_samples, dtype=float)
        compensated_conditions = np.empty(n_samples, dtype=float)
        system_conditions = np.empty(n_samples, dtype=float)
        local_lambdas = np.empty(n_samples, dtype=float)
        covariance_factors = (
            np.empty((n_samples, n_parameters), dtype=float)
            if compute_inference
            else None
        )
        hat_matrix = (
            np.empty((n_samples, n_samples), dtype=float) if store_hat_matrix else None
        )
        trace_sts = 0.0

        for index, distance_row in enumerate(distances):
            weights = self._weights_for_candidate(distance_row, self.bandwidth_)
            n_positive = int(np.count_nonzero(weights > 0.0))
            if n_positive < n_parameters:
                warnings.warn(
                    f"Location {index}: only {n_positive} positive-weight observations "
                    f"are available for {n_parameters} design columns. The local ridge "
                    "term may stabilize the estimate, but a larger bandwidth is advised.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            (
                beta,
                hat_row,
                condition,
                local_lambda,
                compensated,
                system_condition,
            ) = self._fit_one_location(
                X_design,
                self.y_train_,
                weights,
                X_design[index],
                self.design_scales_,
            )
            params[index] = beta
            fitted[index] = float(X_design[index] @ beta)
            influence[index] = float(hat_row[index])
            condition_numbers[index] = condition
            local_lambdas[index] = local_lambda
            compensated_conditions[index] = compensated
            system_conditions[index] = system_condition
            trace_sts += float(np.dot(hat_row, hat_row))
            if hat_matrix is not None:
                hat_matrix[index] = hat_row
            if covariance_factors is not None:
                transform, _ = self._ridge_transform(
                    X_design,
                    weights,
                    self.design_scales_,
                    local_lambda,
                )
                covariance_factors[index] = np.sum(transform**2, axis=1)

        return _LCRLocalFitResult(
            params=params,
            fitted_values=fitted,
            distances=distances,
            influence=influence,
            trace_S=float(np.sum(influence)),
            trace_StS=float(trace_sts),
            covariance_factors=covariance_factors,
            hat_matrix=hat_matrix,
            condition_numbers=condition_numbers,
            compensated_condition_numbers=compensated_conditions,
            penalized_system_condition_numbers=system_conditions,
            local_lambdas=local_lambdas,
        )

    def _cross_validation_residuals(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        bandwidth: Union[int, float],
        design_scales: np.ndarray,
    ) -> np.ndarray:
        residuals = np.empty(y.shape[0], dtype=float)
        for index, distance_row in enumerate(distances):
            weights = self._weights_for_candidate(distance_row, bandwidth).copy()
            weights[index] = 0.0
            try:
                beta, _, _, _, _, _ = self._fit_one_location(
                    X_design,
                    y,
                    weights,
                    X_design[index],
                    design_scales,
                )
                residuals[index] = float(y[index] - X_design[index] @ beta)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                residuals[index] = np.inf
        return residuals

    def _automatic_search_bounds(
        self,
        X_design: np.ndarray,
        distances: np.ndarray,
    ) -> Tuple[float, float]:
        if self.bandwidth_range is not None:
            lower, upper = map(float, self.bandwidth_range)
        elif self.adaptive:
            minimum = X_design.shape[1] + 1
            # GWmodel starts at 20 neighbours. Retain that convention when the
            # dataset is large enough, while allowing deterministic small tests.
            lower = float(max(minimum, 20 if X_design.shape[0] >= 20 else minimum))
            upper = float(X_design.shape[0])
        else:
            upper = float(np.max(distances))
            if upper <= 0.0:
                raise ValueError(
                    "A fixed bandwidth cannot be selected from zero distances."
                )
            lower = upper / 5000.0

        if self.adaptive:
            lower = float(int(np.ceil(lower)))
            upper = float(int(np.floor(upper)))
            minimum = X_design.shape[1] + 1
            lower = max(lower, float(minimum))
            upper = min(upper, float(X_design.shape[0]))
        if lower <= 0.0 or upper < lower:
            raise ValueError(
                f"Invalid LCRGWR bandwidth search interval [{lower}, {upper}]."
            )
        return lower, upper

    def _select_bandwidth_lcr(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        design_scales: np.ndarray,
    ) -> Union[int, float]:
        if not isinstance(self.bandwidth, str) and self.bandwidth is not None:
            return super()._resolve_bandwidth(X_design, y, self.coords_train_)

        method = (
            self.bandwidth.strip().lower()
            if isinstance(self.bandwidth, str)
            else self.bandwidth_method.strip().lower()
        )
        if method != "cv":
            raise ValueError(
                "Classical LCRGWR automatic bandwidth selection supports only "
                "leave-one-out cross-validation ('cv'), matching bw.gwr.lcr."
            )

        lower, upper = self._automatic_search_bounds(X_design, distances)

        def objective(candidate: float) -> float:
            bandwidth = int(round(candidate)) if self.adaptive else float(candidate)
            residuals = self._cross_validation_residuals(
                X_design,
                y,
                distances,
                bandwidth,
                design_scales,
            )
            if not np.all(np.isfinite(residuals)):
                return np.inf
            return float(np.dot(residuals, residuals))

        if self.optimization_method == "grid":
            candidates = (
                np.arange(int(lower), int(upper) + 1, dtype=float)
                if self.adaptive
                else np.linspace(lower, upper, num=31)
            )
            scores = np.array([objective(value) for value in candidates], dtype=float)
            if not np.any(np.isfinite(scores)):
                raise RuntimeError("No finite LCRGWR bandwidth candidate was found.")
            best_index = int(np.nanargmin(scores))
            selected = float(candidates[best_index])
            score = float(scores[best_index])
            self.bandwidth_selection_result_ = {
                "value": int(selected) if self.adaptive else selected,
                "score": score,
                "evaluations": int(candidates.size),
                "converged": True,
                "method": "grid",
            }
        else:
            if self.adaptive or self.optimization_method == "golden_section":
                optimizer = GoldenSectionSearch(
                    tol=1.0e-5,
                    max_iter=100,
                    verbose=self.verbose,
                )
                result = optimizer.minimize(
                    objective,
                    lower,
                    upper,
                    adaptive=self.adaptive,
                )
            elif self.optimization_method == "brent":
                optimizer = BrentSearch(
                    tol=1.0e-5,
                    max_iter=100,
                    verbose=self.verbose,
                )
                result = optimizer.minimize(objective, lower, upper)
            else:
                raise ValueError(
                    "optimization_method must be 'grid', 'golden_section', or 'brent'."
                )
            if not np.isfinite(result.score):
                raise RuntimeError("No finite LCRGWR bandwidth candidate was found.")
            selected = float(result.value)
            score = float(result.score)
            self.bandwidth_selection_result_ = {
                "value": int(round(selected)) if self.adaptive else selected,
                "score": score,
                "evaluations": int(result.evaluations),
                "iterations": int(result.iterations),
                "converged": bool(result.converged),
                "message": result.message,
                "method": self.optimization_method,
            }

        self.bandwidth_cv_score_ = score
        return int(round(selected)) if self.adaptive else selected

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        compute_hat_matrix: bool = True,
        compute_local_r2: bool = True,
        compute_inference: bool = True,
        compute_cv: bool = True,
        verbose: Optional[bool] = None,
    ) -> "LCRGWR":
        """Fit LCR-GWR and return ``self``.

        Args:
            X: Predictor matrix with shape ``(n_samples, n_features)``.
            y: Response vector with shape ``(n_samples,)``.
            coords: Spatial coordinates with shape ``(n_samples, 2)``.
            compute_hat_matrix: Whether to retain the complete penalized smoother
                matrix. Trace statistics are always computed.
            compute_local_r2: Whether to compute local coefficients of determination.
            compute_inference: Whether to compute local standard errors and t values.
            compute_cv: Whether to compute leave-one-out residuals at the final
                bandwidth.
            verbose: Optional per-fit override of the estimator verbosity.

        Returns:
            The fitted estimator.
        """
        for name, value in (
            ("compute_hat_matrix", compute_hat_matrix),
            ("compute_local_r2", compute_local_r2),
            ("compute_inference", compute_inference),
            ("compute_cv", compute_cv),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
        if verbose is not None:
            if not isinstance(verbose, (bool, np.bool_)):
                raise TypeError("verbose must be boolean or None.")
            self.verbose = bool(verbose)

        self._validate_gwr_parameters(
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            bandwidth_method=self.bandwidth_method,
            adaptive=self.adaptive,
            bandwidth_range=self.bandwidth_range,
            optimization_method=self.optimization_method,
        )
        self._validate_lcr_parameters()
        self._reset_fit_state()

        try:
            X_arr, y_arr, coords_arr = self._validate_inputs(X, y, coords)
            feature_names = (
                None
                if self.feature_names_in_ is None
                else self.feature_names_in_.copy()
            )
            self._store_training_data(X_arr, y_arr, coords_arr, copy=True)
            self.feature_names_in_ = feature_names
            X_design = (
                add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
            )
            self.design_scales_ = self._compute_design_scales(X_design)
            self.kernel_func_ = get_kernel_function(self.kernel)
            distances = np.asarray(
                compute_distance_matrix(
                    self.coords_train_,
                    self.coords_train_,
                    metric=self.distance_metric,
                ),
                dtype=float,
            )
            self.bandwidth_ = self._select_bandwidth_lcr(
                X_design,
                self.y_train_,
                distances,
                self.design_scales_,
            )

            if self.verbose:
                bandwidth_kind = (
                    "adaptive neighbours" if self.adaptive else "fixed distance"
                )
                print(
                    f"Fitting LCRGWR with bandwidth={self.bandwidth_} "
                    f"({bandwidth_kind}), lambda_adjust={self.lambda_adjust}, "
                    f"cn_thresh={self.cn_thresh}..."
                )

            self.inference_enabled_ = bool(compute_inference)
            local_fit = self._fit_training_locations_lcr(
                X_design,
                distances,
                store_hat_matrix=bool(compute_hat_matrix),
                compute_inference=self.inference_enabled_,
            )
            if self.fit_intercept:
                self.intercept_ = local_fit.params[:, 0].copy()
                self.coef_ = local_fit.params[:, 1:].copy()
            else:
                self.intercept_ = np.zeros(self.n_samples_, dtype=float)
                self.coef_ = local_fit.params.copy()

            self.coefficients_ = local_fit.params.copy()
            self.fitted_values_ = local_fit.fitted_values.copy()
            self.residuals_ = self.y_train_ - self.fitted_values_
            self.influence_ = local_fit.influence.copy()
            self.hat_matrix_ = local_fit.hat_matrix
            self.S_matrix_ = self.hat_matrix_
            self.condition_numbers_ = local_fit.condition_numbers.copy()
            self.local_condition_numbers_ = self.condition_numbers_
            self.compensated_condition_numbers_ = (
                local_fit.compensated_condition_numbers.copy()
            )
            self.penalized_system_condition_numbers_ = (
                local_fit.penalized_system_condition_numbers.copy()
            )
            self.local_lambda_ = local_fit.local_lambdas.copy()
            self.local_lambdas_ = self.local_lambda_
            self.locally_compensated_mask_ = self.lambda_adjust & (
                self.condition_numbers_ > self.cn_thresh
            )
            self.ridge_applied_mask_ = self.local_lambda_ > 0.0

            self.diagnostics_ = compute_diagnostics(
                self.y_train_,
                self.fitted_values_,
                compute_gwr_stats=True,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )
            self.diagnostics_.update(
                {
                    "mean_condition_number": float(np.mean(self.condition_numbers_)),
                    "max_condition_number": float(np.max(self.condition_numbers_)),
                    "mean_local_lambda": float(np.mean(self.local_lambda_)),
                    "max_local_lambda": float(np.max(self.local_lambda_)),
                    "n_locally_compensated": float(
                        np.count_nonzero(self.locally_compensated_mask_)
                    ),
                }
            )
            self.local_r2_ = (
                self._compute_local_r2_from_distances(distances)
                if compute_local_r2
                else None
            )
            self._set_inference_results(
                local_fit.covariance_factors,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )

            if compute_cv:
                self.cv_residuals_ = self._cross_validation_residuals(
                    X_design,
                    self.y_train_,
                    distances,
                    self.bandwidth_,
                    self.design_scales_,
                )
                self.cv_contributions_ = self.cv_residuals_**2
                self.bandwidth_cv_score_ = float(np.sum(self.cv_contributions_))

            self._mark_fitted()
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def _prediction_parameters(
        self,
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Optional[np.ndarray]]:
        self._check_is_fitted()
        if (
            self.X_train_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self.bandwidth_ is None
            or self.kernel_func_ is None
            or self.design_scales_ is None
        ):
            raise RuntimeError("Stored LCRGWR training state is incomplete.")

        coords_arr = validate_coords(coords)
        X_design = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        distances = compute_distance_matrix(
            coords_arr,
            self.coords_train_,
            metric=self.distance_metric,
        )
        full_params = np.empty((coords_arr.shape[0], X_design.shape[1]), dtype=float)
        covariance_factors = (
            np.empty_like(full_params) if self.inference_enabled_ else None
        )
        conditions = np.empty(coords_arr.shape[0], dtype=float)
        compensated = np.empty(coords_arr.shape[0], dtype=float)
        system_conditions = np.empty(coords_arr.shape[0], dtype=float)
        lambdas = np.empty(coords_arr.shape[0], dtype=float)

        for index, distance_row in enumerate(distances):
            weights = self._weights_for_candidate(distance_row, self.bandwidth_)
            condition, local_lambda, adjusted, _ = self._condition_and_lambda(
                X_design,
                weights,
            )
            transform, system_condition = self._ridge_transform(
                X_design,
                weights,
                self.design_scales_,
                local_lambda,
            )
            full_params[index] = transform @ self.y_train_
            conditions[index] = condition
            compensated[index] = adjusted
            system_conditions[index] = system_condition
            lambdas[index] = local_lambda
            if covariance_factors is not None:
                covariance_factors[index] = np.sum(transform**2, axis=1)

        if self.fit_intercept:
            intercept = full_params[:, 0]
            coef = full_params[:, 1:]
        else:
            intercept = np.zeros(coords_arr.shape[0], dtype=float)
            coef = full_params

        standard_errors: Optional[np.ndarray] = None
        t_values: Optional[np.ndarray] = None
        if (
            covariance_factors is not None
            and self.sigma2_ is not None
            and np.isfinite(self.sigma2_)
        ):
            standard_errors = np.sqrt(
                np.maximum(covariance_factors * self.sigma2_, 0.0)
            )
            t_values = np.full_like(full_params, np.nan, dtype=float)
            np.divide(
                full_params,
                standard_errors,
                out=t_values,
                where=standard_errors > np.finfo(float).eps,
            )

        return {
            "coords": coords_arr,
            "coef": coef,
            "intercept": intercept,
            "standard_errors": standard_errors,
            "t_values": t_values,
            "condition_numbers": conditions,
            "compensated_condition_numbers": compensated,
            "penalized_system_condition_numbers": system_conditions,
            "local_lambdas": lambdas,
        }

    def predict_result(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> GWRPredictionResult:
        """Predict responses and return local parameters and inference results."""
        X_arr, coords_arr = self._validate_prediction_inputs(X, coords)
        params = self._prediction_parameters(coords_arr)
        coef = np.asarray(params["coef"], dtype=float)
        intercept = np.asarray(params["intercept"], dtype=float)
        predictions = np.einsum("ij,ij->i", X_arr, coef) + intercept
        names = (
            tuple(str(name) for name in self.feature_names_in_)
            if self.feature_names_in_ is not None
            else tuple(f"x{index}" for index in range(X_arr.shape[1]))
        )

        full_se = params["standard_errors"]
        full_t = params["t_values"]
        if full_se is not None:
            if self.fit_intercept:
                intercept_se = full_se[:, 0]
                coef_se = full_se[:, 1:]
            else:
                intercept_se = np.zeros(coords_arr.shape[0], dtype=float)
                coef_se = full_se
        else:
            intercept_se = None
            coef_se = None
        if full_t is not None:
            if self.fit_intercept:
                intercept_t = full_t[:, 0]
                coef_t = full_t[:, 1:]
            else:
                intercept_t = np.full(coords_arr.shape[0], np.nan, dtype=float)
                coef_t = full_t
        else:
            intercept_t = None
            coef_t = None

        return GWRPredictionResult(
            predictions=np.asarray(predictions, dtype=float),
            coef=coef,
            intercept=intercept,
            coords=np.asarray(coords_arr, dtype=float),
            feature_names=names,
            coef_standard_errors=coef_se,
            intercept_standard_errors=intercept_se,
            coef_t_values=coef_t,
            intercept_t_values=intercept_t,
        )

    def get_local_diagnostics(
        self,
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> pd.DataFrame:
        """Return condition numbers and ridge parameters at target locations."""
        params = self._prediction_parameters(coords)
        coords_arr = np.asarray(params["coords"], dtype=float)
        return pd.DataFrame(
            {
                "coord_0": coords_arr[:, 0],
                "coord_1": coords_arr[:, 1],
                "condition_number": params["condition_numbers"],
                "local_lambda": params["local_lambdas"],
                "compensated_condition_number": params["compensated_condition_numbers"],
                "penalized_system_condition_number": params[
                    "penalized_system_condition_numbers"
                ],
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Return standard GWR outputs plus LCR diagnostics."""
        frame = super().to_frame()
        if self.condition_numbers_ is not None:
            frame["condition_number"] = self.condition_numbers_
        if self.local_lambda_ is not None:
            frame["local_lambda"] = self.local_lambda_
        if self.compensated_condition_numbers_ is not None:
            frame["compensated_condition_number"] = self.compensated_condition_numbers_
        if self.penalized_system_condition_numbers_ is not None:
            frame["penalized_system_condition_number"] = (
                self.penalized_system_condition_numbers_
            )
        if self.locally_compensated_mask_ is not None:
            frame["locally_compensated"] = self.locally_compensated_mask_
        if self.cv_residuals_ is not None:
            frame["cv_residual"] = self.cv_residuals_
            frame["cv_score"] = self.cv_contributions_
        return frame

    def summary(self) -> str:
        """Return a stable text summary of the LCR-GWR fit."""
        self._check_is_fitted()
        if (
            self.condition_numbers_ is None
            or self.local_lambda_ is None
            or self.locally_compensated_mask_ is None
            or self.diagnostics_ is None
        ):
            raise RuntimeError("LCRGWR diagnostics are unavailable.")

        lines = [
            "=" * 78,
            "Locally Compensated Ridge Geographically Weighted Regression",
            "=" * 78,
            f"Samples: {self.n_samples_}",
            f"Predictors: {self.n_features_in_}",
            f"Kernel: {self.kernel}",
            f"Bandwidth: {self.bandwidth_} "
            f"({'adaptive neighbours' if self.adaptive else 'fixed distance'})",
            f"Distance metric: {self.distance_metric}",
            f"Global ridge lambda: {self.lambda_ridge:.6g}",
            f"Local compensation: {self.lambda_adjust}",
            f"Condition-number threshold: {self.cn_thresh:.6g}",
            f"Locally compensated locations: "
            f"{np.count_nonzero(self.locally_compensated_mask_)}",
            "",
            "Local collinearity and ridge diagnostics",
            "-" * 78,
            f"Condition number min/median/mean/max: "
            f"{np.min(self.condition_numbers_):.4f} / "
            f"{np.median(self.condition_numbers_):.4f} / "
            f"{np.mean(self.condition_numbers_):.4f} / "
            f"{np.max(self.condition_numbers_):.4f}",
            f"Local lambda min/median/mean/max: "
            f"{np.min(self.local_lambda_):.6g} / "
            f"{np.median(self.local_lambda_):.6g} / "
            f"{np.mean(self.local_lambda_):.6g} / "
            f"{np.max(self.local_lambda_):.6g}",
            "",
            "Model diagnostics",
            "-" * 78,
            f"RSS: {self.diagnostics_['rss']:.6f}",
            f"R-squared: {self.diagnostics_['r2']:.6f}",
            f"Adjusted R-squared: {self.diagnostics_['adj_r2']:.6f}",
            f"trace(S): {self.diagnostics_['trace_S']:.6f}",
            f"trace(S'S): {self.diagnostics_['trace_StS']:.6f}",
            f"AIC: {self.diagnostics_['aic']:.6f}",
            f"AICc: {self.diagnostics_['aicc']:.6f}",
            f"BIC: {self.diagnostics_['bic']:.6f}",
        ]
        if self.bandwidth_cv_score_ is not None:
            lines.append(f"Leave-one-out CV score: {self.bandwidth_cv_score_:.6f}")
        lines.append("=" * 78)
        return "\n".join(lines)
