# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Robust geographically weighted regression.

This module implements the two classical robust GWR procedures exposed by
``GWmodel::gwr.robust``: iterative automatic residual reweighting and a
single filtered-outlier refit. The estimator reuses the validated Gaussian
GWR calibration, inference, prediction, and result interfaces from
:class:`pygwrx.models.GWR`.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.utils import add_intercept
from pygwrx.models.gwr import GWR


class RGWR(GWR):
    """Classical robust geographically weighted regression.

    RGWR first calibrates a standard Gaussian GWR using the requested kernel
    and bandwidth. It then applies one of the robust procedures implemented by
    the R package ``GWmodel``:

    ``"automatic"``
        Repeatedly combine the spatial kernel weights with a global residual
        weight vector. Standardized residual magnitudes below ``cut1`` receive
        weight 1, values between ``cut1`` and ``cut2`` receive a smooth
        bisquare transition, and values above ``cut2`` receive weight 0.

    ``"filtered"``
        Compute GWmodel-style studentized residuals from the initial GWR hat
        matrix, exclude observations whose absolute residual exceeds
        ``cut_filter``, and refit once.

    Args:
        kernel: Spatial kernel name or callable accepted by :class:`GWR`.
        bandwidth: Numeric bandwidth or automatic-selection criterion. Robust
            fitting uses the bandwidth selected by the initial standard GWR.
        bandwidth_method: Criterion used when ``bandwidth=None``.
        adaptive: Whether bandwidths represent nearest-neighbour counts.
        bandwidth_range: Optional lower and upper bandwidth search bounds.
        optimization_method: One-dimensional bandwidth search method.
        fit_intercept: Whether to include a local intercept.
        distance_metric: Distance metric used by the spatial kernel.
        sigma2_v1: Residual-variance convention used for final GWR inference.
        method: Robust procedure, either ``"automatic"`` or ``"filtered"``.
        max_iter: Maximum number of automatic robust refits.
        tol: Relative mean-squared-error tolerance for automatic convergence.
        cut1: Lower standardized-residual threshold for automatic reweighting.
        cut2: Upper standardized-residual threshold for automatic reweighting.
        cut_filter: Absolute studentized-residual threshold for filtered RGWR.
        verbose: Whether to print fit progress.

    Attributes:
        robust_weights_: Observation-level residual weights used in the final
            robust refit. Filtered weights are exactly 0 or 1.
        outlier_mask_: Boolean mask identifying zero-weight observations.
        downweighted_mask_: Boolean mask identifying observations with final
            robust weights below 1.
        n_iter_: Number of robust refits after the initial standard GWR.
        converged_: Whether the automatic relative-MSE criterion was reached.
            Filtered RGWR is marked converged after its single refit.
        weight_history_: Robust weight vectors used by successive calibrations,
            beginning with the all-ones initial GWR weights.
        mse_history_: Initial and robust-refit mean squared residuals.
        convergence_history_: Relative MSE changes for automatic RGWR.
        initial_studentized_residuals_: GWmodel-style studentized residuals
            from the initial standard GWR. Populated for filtered RGWR.
        robust_residual_scores_: Final residuals divided by the root mean
            squared residual, matching the automatic weight-score definition.

    Notes:
        The robust weights are observation-level weights shared by every local
        calibration. At location :math:`s_i`, the effective weights are

        .. math::

            w_{ij}^{\\mathrm{effective}}
            = w_{ij}^{\\mathrm{spatial}} r_j,

        where :math:`r_j` is the final residual weight for observation ``j``.

        Bandwidth selection is intentionally performed on the initial standard
        GWR, matching the standard GWmodel workflow in which ``gwr.robust`` is
        supplied a bandwidth selected by ``bw.gwr``.

    References:
        Harris, P., Fotheringham, A. S., and Juggins, S. (2010). Robust
        geographically weighted regression: a technique for quantifying
        spatial relationships between freshwater acidification critical loads
        and catchment attributes. *Annals of the Association of American
        Geographers*, 100(2), 286-306.

        Lu, B., Harris, P., Charlton, M., and Brunsdon, C. (2014). The GWmodel
        R package: further topics for exploring spatial heterogeneity using
        geographically weighted models. *Geo-spatial Information Science*,
        17(2), 85-101.
    """

    _VALID_METHODS = {"automatic", "filtered"}

    def __init__(
        self,
        kernel: Union[str, Callable[[np.ndarray, float], np.ndarray]] = "gaussian",
        bandwidth: Union[float, int, str, None] = "cv",
        bandwidth_method: str = "cv",
        adaptive: bool = False,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        optimization_method: str = "golden_section",
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        method: str = "automatic",
        max_iter: int = 20,
        tol: float = 1.0e-5,
        cut1: float = 2.0,
        cut2: float = 3.0,
        cut_filter: float = 3.0,
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
        self.method = self._normalize_method(method)
        self.max_iter = self._validate_positive_integer(max_iter, "max_iter")
        self.tol = self._validate_positive_float(tol, "tol")
        self.cut1 = self._validate_nonnegative_float(cut1, "cut1")
        self.cut2 = self._validate_positive_float(cut2, "cut2")
        self.cut_filter = self._validate_positive_float(cut_filter, "cut_filter")
        if self.cut2 <= self.cut1:
            raise ValueError("cut2 must be greater than cut1.")
        self._reset_robust_state()

    @classmethod
    def _normalize_method(cls, method: str) -> str:
        if not isinstance(method, str) or not method.strip():
            raise TypeError("method must be a non-empty string.")
        normalized = method.strip().lower()
        if normalized not in cls._VALID_METHODS:
            raise ValueError(
                "method must be one of "
                f"{sorted(cls._VALID_METHODS)}; received {method!r}."
            )
        return normalized

    @staticmethod
    def _validate_positive_integer(value: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
        result = int(value)
        if result < 1:
            raise ValueError(f"{name} must be at least 1.")
        return result

    @staticmethod
    def _validate_positive_float(value: float, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric.")
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero.")
        return result

    @staticmethod
    def _validate_nonnegative_float(value: float, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric.")
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return result

    def _validate_robust_parameters(self) -> None:
        self.method = self._normalize_method(self.method)
        self.max_iter = self._validate_positive_integer(self.max_iter, "max_iter")
        self.tol = self._validate_positive_float(self.tol, "tol")
        self.cut1 = self._validate_nonnegative_float(self.cut1, "cut1")
        self.cut2 = self._validate_positive_float(self.cut2, "cut2")
        self.cut_filter = self._validate_positive_float(self.cut_filter, "cut_filter")
        if self.cut2 <= self.cut1:
            raise ValueError("cut2 must be greater than cut1.")

    def _reset_robust_state(self) -> None:
        self.robust_method_: Optional[str] = None
        self.robust_weights_: Optional[np.ndarray] = None
        self.outlier_mask_: Optional[np.ndarray] = None
        self.downweighted_mask_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.weight_history_: List[np.ndarray] = []
        self.mse_history_: List[float] = []
        self.convergence_history_: List[float] = []
        self.initial_fitted_values_: Optional[np.ndarray] = None
        self.initial_residuals_: Optional[np.ndarray] = None
        self.initial_diagnostics_: Optional[dict] = None
        self.initial_studentized_residuals_: Optional[np.ndarray] = None
        self.robust_residual_scores_: Optional[np.ndarray] = None
        self.robust_scale_: Optional[float] = None

    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self._reset_robust_state()

    def _weights_from_distances(self, distances: np.ndarray) -> np.ndarray:
        spatial_weights = super()._weights_from_distances(distances)
        if self.robust_weights_ is None:
            return spatial_weights

        robust_weights = np.asarray(self.robust_weights_, dtype=float)
        if robust_weights.shape != spatial_weights.shape:
            raise RuntimeError(
                "robust_weights_ must match the number of calibration observations."
            )
        total_weights = spatial_weights * robust_weights
        if not np.all(np.isfinite(total_weights)) or np.any(total_weights < 0.0):
            raise RuntimeError("The combined spatial and robust weights are invalid.")
        if not np.any(total_weights > 0.0):
            raise RuntimeError(
                "The combined spatial and robust weights contain no positive values."
            )
        return total_weights

    def _automatic_weights(
        self, residuals: np.ndarray, mse: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        if not np.isfinite(mse) or mse < 0.0:
            raise RuntimeError(
                "Residual mean squared error must be finite and non-negative."
            )

        if mse <= np.finfo(float).eps:
            scores = np.zeros_like(residuals)
        else:
            scores = np.abs(residuals / np.sqrt(mse))

        weights = np.ones_like(scores)
        transition = scores > self.cut1
        rejected = scores > self.cut2
        span = self.cut2 - self.cut1
        weights[transition] = (
            1.0 - ((scores[transition] - self.cut1) / span) ** 2
        ) ** 2
        weights[rejected] = 0.0
        weights[~np.isfinite(weights)] = 0.0
        return weights, scores

    def _gwmodel_studentized_residuals(self) -> np.ndarray:
        if self.hat_matrix_ is None or self.residuals_ is None:
            raise RuntimeError(
                "Filtered RGWR requires the initial full GWR hat matrix."
            )
        if self.diagnostics_ is None:
            raise RuntimeError("Initial GWR diagnostics are unavailable.")

        hat_matrix = np.asarray(self.hat_matrix_, dtype=float)
        identity_minus_s = np.eye(hat_matrix.shape[0], dtype=float) - hat_matrix
        # GWmodel accumulates squared entries of each (I - S) row into a
        # column-wise q diagonal before constructing studentized residuals.
        q_diagonal = np.sum(identity_minus_s**2, axis=0)
        trace_s = float(self.diagnostics_.get("trace_S", np.nan))
        trace_sts = float(self.diagnostics_.get("trace_StS", np.nan))
        rss = float(np.dot(self.residuals_, self.residuals_))
        denominator = self.n_samples_ - 2.0 * trace_s + trace_sts
        sigma2 = rss / denominator if denominator > 0.0 else np.nan

        scale_squared = sigma2 * q_diagonal
        studentized = np.full(self.n_samples_, np.nan, dtype=float)
        valid = np.isfinite(scale_squared) & (scale_squared > np.finfo(float).eps)
        studentized[valid] = self.residuals_[valid] / np.sqrt(scale_squared[valid])
        exact_zero = (~valid) & np.isclose(
            self.residuals_, 0.0, rtol=0.0, atol=np.finfo(float).eps
        )
        studentized[exact_zero] = 0.0
        return studentized

    def _ensure_effective_sample(self, weights: np.ndarray, n_parameters: int) -> None:
        n_positive = int(np.count_nonzero(np.asarray(weights) > 0.0))
        if n_positive < n_parameters:
            raise RuntimeError(
                "Robust reweighting retained only "
                f"{n_positive} positive-weight observations for {n_parameters} "
                "design columns. Increase the bandwidth or relax the robust "
                "residual thresholds."
            )

    def _commit_robust_fit(
        self,
        X_design: np.ndarray,
        *,
        compute_hat_matrix: bool,
        compute_local_r2: bool,
        compute_inference: bool,
    ) -> None:
        self._ensure_effective_sample(self.robust_weights_, X_design.shape[1])
        self._reset_inference_state()
        self.inference_enabled_ = bool(compute_inference)
        local_fit = self._fit_training_locations(
            X_design,
            store_hat_matrix=bool(compute_hat_matrix),
            compute_inference=self.inference_enabled_,
        )

        if self.fit_intercept:
            self.intercept_ = local_fit.params[:, 0].copy()
            self.coef_ = local_fit.params[:, 1:].copy()
        else:
            self.intercept_ = np.zeros(self.n_samples_, dtype=float)
            self.coef_ = local_fit.params.copy()

        self.fitted_values_ = local_fit.fitted_values.copy()
        self.residuals_ = self.y_train_ - self.fitted_values_
        self.influence_ = local_fit.influence.copy()
        self.hat_matrix_ = local_fit.hat_matrix
        self.S_matrix_ = self.hat_matrix_
        self.diagnostics_ = compute_diagnostics(
            self.y_train_,
            self.fitted_values_,
            compute_gwr_stats=True,
            trace_S=local_fit.trace_S,
            trace_StS=local_fit.trace_StS,
        )

        if compute_local_r2:
            # GWmodel reports local R² using the spatial kernel alone rather
            # than multiplying it by the residual robustness vector.
            saved_weights = self.robust_weights_
            self.robust_weights_ = None
            try:
                self.local_r2_ = self._compute_local_r2_from_distances(
                    local_fit.distances
                )
            finally:
                self.robust_weights_ = saved_weights
        else:
            self.local_r2_ = None

        self._set_inference_results(
            local_fit.covariance_factors,
            trace_S=local_fit.trace_S,
            trace_StS=local_fit.trace_StS,
        )
        self._mark_fitted()

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        compute_hat_matrix: bool = True,
        compute_local_r2: bool = True,
        compute_inference: bool = True,
        compute_hat_matrix_flag: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> "RGWR":
        """Fit robust GWR and return the estimator.

        Args:
            X: Predictor matrix with shape ``(n_samples, n_features)``.
            y: Response vector with shape ``(n_samples,)``.
            coords: Coordinates with shape ``(n_samples, 2)``.
            compute_hat_matrix: Whether to retain the final robust hat matrix.
            compute_local_r2: Whether to compute local coefficients of
                determination.
            compute_inference: Whether to retain parameter covariance factors,
                standard errors, and t values.
            compute_hat_matrix_flag: Compatibility alias for
                ``compute_hat_matrix``.
            verbose: Optional per-fit override of the estimator verbosity.

        Returns:
            The fitted robust estimator.

        Raises:
            RuntimeError: If robust filtering leaves too few usable
                observations for local calibration.
        """
        self._reset_fit_state()
        self._validate_robust_parameters()
        if compute_hat_matrix_flag is not None:
            if not isinstance(compute_hat_matrix_flag, (bool, np.bool_)):
                raise TypeError("compute_hat_matrix_flag must be boolean or None.")
            compute_hat_matrix = bool(compute_hat_matrix_flag)
        for name, value in (
            ("compute_hat_matrix", compute_hat_matrix),
            ("compute_local_r2", compute_local_r2),
            ("compute_inference", compute_inference),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
        try:
            # Filtered RGWR needs the complete initial S matrix to reproduce the
            # studentized residual definition in GWmodel::gwr.basic.
            initial_hat_matrix = bool(compute_hat_matrix) or self.method == "filtered"
            super().fit(
                X,
                y,
                coords,
                compute_hat_matrix=initial_hat_matrix,
                compute_local_r2=bool(compute_local_r2),
                compute_inference=bool(compute_inference),
                compute_hat_matrix_flag=None,
                verbose=verbose,
            )

            self.initial_fitted_values_ = self.fitted_values_.copy()
            self.initial_residuals_ = self.residuals_.copy()
            self.initial_diagnostics_ = dict(self.diagnostics_ or {})
            initial_mse = float(np.mean(self.initial_residuals_**2))
            self.mse_history_ = [initial_mse]
            self.weight_history_ = [np.ones(self.n_samples_, dtype=float)]

            X_design = (
                add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
            )

            if self.method == "filtered":
                studentized = self._gwmodel_studentized_residuals()
                self.initial_studentized_residuals_ = studentized.copy()
                robust_weights = (
                    np.isfinite(studentized) & (np.abs(studentized) < self.cut_filter)
                ).astype(float)
                self.robust_weights_ = robust_weights
                self.weight_history_.append(robust_weights.copy())
                self._commit_robust_fit(
                    X_design,
                    compute_hat_matrix=bool(compute_hat_matrix),
                    compute_local_r2=bool(compute_local_r2),
                    compute_inference=bool(compute_inference),
                )
                final_mse = float(np.mean(self.residuals_**2))
                self.mse_history_.append(final_mse)
                self.n_iter_ = 1
                self.converged_ = True
            else:
                current_mse = initial_mse
                candidate_weights, scores = self._automatic_weights(
                    self.initial_residuals_, current_mse
                )
                self.robust_residual_scores_ = scores.copy()

                for iteration in range(self.max_iter):
                    self.robust_weights_ = candidate_weights.copy()
                    self.weight_history_.append(self.robust_weights_.copy())
                    self._commit_robust_fit(
                        X_design,
                        compute_hat_matrix=bool(compute_hat_matrix),
                        compute_local_r2=bool(compute_local_r2),
                        compute_inference=bool(compute_inference),
                    )

                    new_mse = float(np.mean(self.residuals_**2))
                    self.mse_history_.append(new_mse)
                    if new_mse <= np.finfo(float).eps:
                        relative_change = 0.0
                    else:
                        relative_change = abs(current_mse - new_mse) / new_mse
                    self.convergence_history_.append(float(relative_change))
                    self.n_iter_ = iteration + 1

                    next_weights, scores = self._automatic_weights(
                        self.residuals_, new_mse
                    )
                    self.robust_residual_scores_ = scores.copy()
                    self.robust_scale_ = float(np.sqrt(max(new_mse, 0.0)))

                    if self.verbose:
                        print(
                            "RGWR automatic iteration "
                            f"{self.n_iter_}: relative MSE change="
                            f"{relative_change:.6g}"
                        )
                    if relative_change <= self.tol:
                        self.converged_ = True
                        break

                    current_mse = new_mse
                    candidate_weights = next_weights

                if not self.converged_:
                    warnings.warn(
                        "RGWR automatic reweighting reached max_iter before the "
                        "relative-MSE tolerance was satisfied.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            if self.robust_weights_ is None:
                raise RuntimeError("The final robust weight vector is unavailable.")
            self.outlier_mask_ = np.isclose(self.robust_weights_, 0.0)
            self.downweighted_mask_ = self.robust_weights_ < 1.0
            final_mse = float(np.mean(self.residuals_**2))
            self.robust_scale_ = float(np.sqrt(max(final_mse, 0.0)))
            if final_mse <= np.finfo(float).eps:
                self.robust_residual_scores_ = np.zeros(self.n_samples_, dtype=float)
            else:
                self.robust_residual_scores_ = self.residuals_ / np.sqrt(final_mse)
            self.robust_method_ = self.method
            self._mark_fitted()
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def to_frame(self) -> pd.DataFrame:
        """Return standard GWR results plus robust diagnostics."""
        frame = super().to_frame()
        if self.robust_weights_ is not None:
            frame["robust_weight"] = self.robust_weights_
            frame["downweighted"] = self.downweighted_mask_
            frame["robust_outlier"] = self.outlier_mask_
        if self.robust_residual_scores_ is not None:
            frame["robust_residual_score"] = self.robust_residual_scores_
        if self.initial_studentized_residuals_ is not None:
            frame["initial_studentized_residual"] = self.initial_studentized_residuals_
        return frame

    def summary(self) -> str:
        """Return the standard GWR summary with robust-fit information."""
        base_summary = super().summary()
        lines = base_summary.splitlines()
        for index, line in enumerate(lines):
            if "Gaussian Geographically Weighted Regression" in line:
                lines[index] = "Robust Geographically Weighted Regression (RGWR)"
                break

        if lines and set(lines[-1]) == {"="}:
            closing = lines.pop()
        else:
            closing = "=" * 78
        lines.extend(
            [
                "",
                "Robust calibration",
                "-" * 78,
                f"Method: {self.method}",
                f"Robust refits: {self.n_iter_}",
                f"Converged: {self.converged_}",
                f"Downweighted observations: "
                f"{int(np.count_nonzero(self.downweighted_mask_))}",
                f"Zero-weight outliers: {int(np.count_nonzero(self.outlier_mask_))}",
                f"Minimum robust weight: {float(np.min(self.robust_weights_)):.6f}",
                closing,
            ]
        )
        return "\n".join(lines)


__all__ = ["RGWR"]
