# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Robust geographically weighted regression.

This module implements the two classical robust GWR procedures exposed by
``GWmodel::gwr.robust``: iterative automatic residual reweighting and a
single filtered-outlier refit. The estimator reuses the validated private standard-GWR execution engine
for calibration, inference, and prediction while owning its estimator
lifecycle and robust result interface directly.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx.core.base import BaseSpatialRegressor
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.solver import _weighted_least_squares_details
from pygwrx.core.utils import _iter_distance_rows as _iter_core_distance_rows
from pygwrx.core.utils import add_intercept, validate_coords
from pygwrx.models._gwr_engine import (
    _collect_gwr_inference,
    _compute_gwr_local_r2,
    _fit_gwr_prediction_locations,
    _fit_gwr_training_locations,
    _get_gwr_bandwidth_selector,
    _gwr_spatial_weights,
    _GWRLocalFitResult,
)
from pygwrx.models.gwr import GWRPredictionResult


class RGWR(BaseSpatialRegressor):
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
        if not isinstance(sigma2_v1, (bool, np.bool_)):
            raise TypeError("sigma2_v1 must be boolean.")
        if isinstance(bandwidth, str) and bandwidth.strip().lower() == "adaptive":
            raise ValueError(
                "GWR uses adaptive=True to request a nearest-neighbour bandwidth; "
                "bandwidth must be numeric, None, or one of 'cv', 'aic', 'aicc', 'bic'."
            )
        super().__init__(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method=bandwidth_method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            verbose=verbose,
        )
        self.sigma2_v1 = bool(sigma2_v1)
        self.S_matrix_: Optional[np.ndarray] = None
        self.bandwidth_search_: Optional[Dict[str, object]] = None
        self._reset_inference_state()
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

    def _reset_inference_state(self) -> None:
        self.sigma2_: Optional[float] = None
        self.influence_: Optional[np.ndarray] = None
        self.standardized_residuals_: Optional[np.ndarray] = None
        self.cooks_distance_: Optional[np.ndarray] = None
        self.parameter_covariance_diagonal_: Optional[np.ndarray] = None
        self.parameter_standard_errors_: Optional[np.ndarray] = None
        self.parameter_t_values_: Optional[np.ndarray] = None
        self.intercept_se_: Optional[np.ndarray] = None
        self.coef_se_: Optional[np.ndarray] = None
        self.intercept_t_: Optional[np.ndarray] = None
        self.coef_t_: Optional[np.ndarray] = None
        self.local_rank_: Optional[np.ndarray] = None
        self.local_condition_number_: Optional[np.ndarray] = None
        self.rank_deficient_: Optional[np.ndarray] = None
        self.inference_enabled_: bool = False

    def _resolve_bandwidth(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
    ) -> Union[int, float]:
        if isinstance(self.bandwidth, str) or self.bandwidth is None:
            method = (
                self.bandwidth.strip().lower()
                if isinstance(self.bandwidth, str)
                else self.bandwidth_method.strip().lower()
            )
            if method not in {"cv", "aic", "aicc", "bic"}:
                raise ValueError(
                    "GWR bandwidth selection method must be one of "
                    "'cv', 'aic', 'aicc', or 'bic'."
                )
            selector = _get_gwr_bandwidth_selector(
                method,
                adaptive=self.adaptive,
                verbose=self.verbose,
                optimization_method=self.optimization_method,
            )
            selected = selector.select(
                X_design,
                y,
                coords,
                self.kernel_func_,
                bandwidth_range=self.bandwidth_range,
                distance_metric=self.distance_metric,
            )
            selected_value: Union[int, float] = (
                int(selected) if self.adaptive else float(selected)
            )
            raw_range = getattr(selector, "search_range_", None)
            if raw_range is None:
                search_range = None
                boundary_solution = False
            elif self.adaptive:
                search_range = (int(raw_range[0]), int(raw_range[1]))
                boundary_solution = int(selected_value) in search_range
            else:
                search_range = (float(raw_range[0]), float(raw_range[1]))
                selected_float = float(selected_value)
                scale = max(
                    1.0,
                    abs(selected_float),
                    abs(search_range[0]),
                    abs(search_range[1]),
                )
                atol = 32.0 * np.finfo(float).eps * scale
                boundary_solution = bool(
                    np.isclose(
                        selected_float,
                        search_range[0],
                        rtol=1e-10,
                        atol=atol,
                    )
                    or np.isclose(
                        selected_float,
                        search_range[1],
                        rtol=1e-10,
                        atol=atol,
                    )
                )

            trace = tuple(getattr(selector, "search_trace_", ()))
            best_score = getattr(selector, "best_score_", None)
            selector_method = getattr(
                selector, "optimization_method", self.optimization_method
            )
            self.bandwidth_search_ = {
                "criterion": method,
                "adaptive": bool(self.adaptive),
                "optimization_method": (
                    "exhaustive_integer" if self.adaptive else selector_method
                ),
                "search_range": search_range,
                "selected": selected_value,
                "best_score": None if best_score is None else float(best_score),
                "trace": trace,
                "boundary_solution": bool(boundary_solution),
            }
            return selected_value

        value = float(self.bandwidth)
        if self.adaptive:
            if not value.is_integer():
                raise ValueError(
                    "adaptive bandwidth must be an integer neighbour count."
                )
            k = int(value)
            minimum = X_design.shape[1] + 1
            if k < minimum:
                raise ValueError(
                    f"adaptive bandwidth must be at least {minimum} for a design "
                    f"matrix with {X_design.shape[1]} columns."
                )
            if k > X_design.shape[0]:
                raise ValueError(
                    f"adaptive bandwidth cannot exceed n_samples={X_design.shape[0]}."
                )
            return k
        return value

    def _iter_distance_rows(self, target_coords: np.ndarray) -> Iterator[np.ndarray]:
        """Yield target-to-training distance rows from the shared bounded backend."""
        if self.coords_train_ is None:
            raise RuntimeError("Training coordinates are unavailable.")
        targets = np.asarray(target_coords, dtype=float)
        if targets.ndim != 2:
            raise ValueError("target_coords must be a two-dimensional array.")
        return _iter_core_distance_rows(
            targets,
            self.coords_train_,
            distance_metric=self.distance_metric,
        )

    @staticmethod
    def _warn_rank_deficiency(
        rank_deficient: np.ndarray,
        *,
        context: str,
        n_parameters: int,
    ) -> None:
        indices = np.flatnonzero(rank_deficient)
        if indices.size == 0:
            return
        preview = ", ".join(str(int(index)) for index in indices[:5])
        suffix = ", ..." if indices.size > 5 else ""
        warnings.warn(
            f"{indices.size} {context} weighted design(s) are rank deficient for "
            f"{n_parameters} parameters (locations: {preview}{suffix}). Coefficients "
            "use the Moore-Penrose minimum-norm WLS solution; coefficient standard "
            "errors and t values are unavailable at rank-deficient locations.",
            RuntimeWarning,
            stacklevel=4,
        )

    def _fit_training_locations(
        self,
        X_design: np.ndarray,
        *,
        store_hat_matrix: bool,
        compute_inference: bool,
    ) -> _GWRLocalFitResult:
        if self.coords_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data are unavailable.")
        return _fit_gwr_training_locations(
            X_design,
            self.y_train_,
            self.coords_train_,
            distance_rows=self._iter_distance_rows,
            weights_from_distances=self._weights_from_distances,
            rank_policy=self._warn_rank_deficiency,
            store_hat_matrix=store_hat_matrix,
            compute_inference=compute_inference,
        )

    def _compute_local_r2_from_distance_rows(
        self, distance_rows: Iterable[np.ndarray]
    ) -> np.ndarray:
        if self.y_train_ is None or self.residuals_ is None:
            raise RuntimeError("Fitted values and residuals are unavailable.")
        return _compute_gwr_local_r2(
            self.y_train_,
            self.residuals_,
            distance_rows,
            weights_from_distances=self._weights_from_distances,
        )

    def _compute_local_r2_from_distances(
        self, distances: Iterable[np.ndarray]
    ) -> np.ndarray:
        """Compatibility helper accepting an array or lazy distance-row iterable."""
        return self._compute_local_r2_from_distance_rows(distances)

    def _compute_local_r2(self) -> np.ndarray:
        if self.coords_train_ is None:
            raise RuntimeError("Training coordinates are unavailable.")
        return self._compute_local_r2_from_distance_rows(
            self._iter_distance_rows(self.coords_train_)
        )

    def _set_inference_results(
        self,
        covariance_factors: Optional[np.ndarray],
        *,
        trace_S: float,
        trace_StS: float,
    ) -> None:
        if self.residuals_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Fitted regression results are unavailable.")

        inference = _collect_gwr_inference(
            self.residuals_,
            self.influence_,
            self.coef_,
            self.intercept_,
            covariance_factors,
            n_samples=self.n_samples_,
            fit_intercept=self.fit_intercept,
            sigma2_v1=self.sigma2_v1,
            trace_S=trace_S,
            trace_StS=trace_StS,
        )
        self.influence_ = inference.influence
        self.sigma2_ = inference.sigma2
        self.standardized_residuals_ = inference.standardized_residuals
        self.cooks_distance_ = inference.cooks_distance
        self.parameter_covariance_diagonal_ = inference.parameter_covariance_diagonal
        self.parameter_standard_errors_ = inference.parameter_standard_errors
        self.parameter_t_values_ = inference.parameter_t_values
        self.intercept_se_ = inference.intercept_se
        self.coef_se_ = inference.coef_se
        self.intercept_t_ = inference.intercept_t
        self.coef_t_ = inference.coef_t

    def _fit_initial_gwr(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        compute_hat_matrix: bool = False,
        compute_local_r2: bool = True,
        compute_inference: bool = True,
        compute_hat_matrix_flag: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> "RGWR":
        """Fit the Gaussian GWR model and return ``self``.

        The smoother traces and influence values are always computed. Setting
        ``compute_hat_matrix=False`` avoids storing the full ``n x n`` smoother
        matrix. Calibration distances are evaluated in bounded row blocks, so a
        numeric-bandwidth fit does not also retain an ``n x n`` distance matrix.
        Automatic bandwidth selection uses the same bounded distance backend.

        ``compute_hat_matrix_flag`` is retained as a compatibility alias for older
        PyGWRx code. New code should use ``compute_hat_matrix``.
        """
        self._reset_fit_state()
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
        if not isinstance(self.sigma2_v1, (bool, np.bool_)):
            raise TypeError("sigma2_v1 must be boolean.")

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
            self.kernel_func_ = get_kernel_function(self.kernel)
            self.bandwidth_ = self._resolve_bandwidth(
                X_design, self.y_train_, self.coords_train_
            )

            if self.verbose:
                kind = "adaptive k" if self.adaptive else "fixed distance"
                print(f"Fitting GWR with {kind} bandwidth={self.bandwidth_}...")

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
            self.local_rank_ = local_fit.local_rank.copy()
            self.local_condition_number_ = local_fit.local_condition_number.copy()
            self.rank_deficient_ = self.local_rank_ < X_design.shape[1]
            self.hat_matrix_ = local_fit.hat_matrix
            self.S_matrix_ = self.hat_matrix_  # compatibility alias
            self.diagnostics_ = compute_diagnostics(
                self.y_train_,
                self.fitted_values_,
                compute_gwr_stats=True,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )

            self.local_r2_ = self._compute_local_r2() if compute_local_r2 else None
            self._set_inference_results(
                local_fit.covariance_factors,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )

            self._mark_fitted()
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        return self.predict_result(X, coords).predictions

    def _prediction_parameters(
        self, coords: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, Optional[np.ndarray]]:
        self._check_is_fitted()
        if (
            self.X_train_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self.bandwidth_ is None
            or self.kernel_func_ is None
        ):
            raise RuntimeError("Stored training state is incomplete.")

        coords_arr = validate_coords(coords)
        X_design = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        local_fit = _fit_gwr_prediction_locations(
            X_design,
            self.y_train_,
            coords_arr,
            distance_rows=self._iter_distance_rows,
            weights_from_distances=self._weights_from_distances,
            rank_policy=self._warn_rank_deficiency,
            compute_inference=self.inference_enabled_,
        )
        full_params = local_fit.full_params
        covariance_factors = local_fit.covariance_factors

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
            "local_rank": local_fit.local_rank,
            "local_condition_number": local_fit.local_condition_number,
            "rank_deficient": local_fit.rank_deficient,
        }

    def predict_result(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> GWRPredictionResult:
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
            local_rank=np.asarray(params["local_rank"], dtype=int),
            local_condition_number=np.asarray(
                params["local_condition_number"], dtype=float
            ),
            rank_deficient=np.asarray(params["rank_deficient"], dtype=bool),
        )

    def get_local_parameters(
        self, coords: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Return local intercepts and slopes at arbitrary coordinates."""
        params = self._prediction_parameters(coords)
        return {
            "intercept": np.asarray(params["intercept"], dtype=float).copy(),
            "coef": np.asarray(params["coef"], dtype=float).copy(),
            "coords": np.asarray(params["coords"], dtype=float).copy(),
            "local_rank": np.asarray(params["local_rank"], dtype=int).copy(),
            "local_condition_number": np.asarray(
                params["local_condition_number"],
                dtype=float,
            ).copy(),
            "rank_deficient": np.asarray(
                params["rank_deficient"],
                dtype=bool,
            ).copy(),
        }

    def get_local_coefficients(
        self, coords: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compatibility helper returning slopes only."""
        return self.get_local_parameters(coords)["coef"]

    def _gwr_to_frame(self) -> pd.DataFrame:
        frame = super().to_frame()
        if self.intercept_se_ is not None:
            frame["intercept_se"] = self.intercept_se_
        if self.intercept_t_ is not None:
            frame["intercept_t"] = self.intercept_t_
        feature_names = (
            [str(name) for name in self.feature_names_in_]
            if self.feature_names_in_ is not None
            else [f"x{index}" for index in range(self.n_features_in_ or 0)]
        )
        if self.coef_se_ is not None:
            for index, name in enumerate(feature_names):
                frame[f"se_{name}"] = self.coef_se_[:, index]
        if self.coef_t_ is not None:
            for index, name in enumerate(feature_names):
                frame[f"t_{name}"] = self.coef_t_[:, index]
        for name, values in (
            ("influence", self.influence_),
            ("standardized_residual", self.standardized_residuals_),
            ("cooks_distance", self.cooks_distance_),
            ("local_rank", self.local_rank_),
            ("local_condition_number", self.local_condition_number_),
            ("rank_deficient", self.rank_deficient_),
        ):
            if values is not None:
                frame[name] = values
        return frame

    def _gwr_summary(self) -> str:
        """Return a stable text summary of global and local model results."""
        self._check_is_fitted()
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data are unavailable.")

        X_global = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        n, p = X_global.shape
        global_solve = _weighted_least_squares_details(
            X_global,
            self.y_train_,
            np.ones(n, dtype=float),
        )
        global_beta = global_solve.beta
        global_fitted = X_global @ global_beta
        global_residuals = self.y_train_ - global_fitted
        global_rss = float(np.dot(global_residuals, global_residuals))
        global_df = max(n - global_solve.rank, 1)
        global_sigma2 = global_rss / global_df
        if global_solve.rank < p:
            global_se = np.full(p, np.nan, dtype=float)
        else:
            global_se = np.sqrt(
                np.maximum(
                    np.diag(global_solve.inverse_normal) * global_sigma2,
                    0.0,
                )
            )

        feature_names = (
            [str(name) for name in self.feature_names_in_]
            if self.feature_names_in_ is not None
            else [f"x{index}" for index in range(self.X_train_.shape[1])]
        )
        global_names = (["intercept"] if self.fit_intercept else []) + feature_names
        local_matrix = (
            np.column_stack([self.intercept_, self.coef_])
            if self.fit_intercept
            else np.asarray(self.coef_)
        )

        bandwidth_lines = [
            f"Bandwidth: {self.bandwidth_} ({'adaptive neighbours' if self.adaptive else 'fixed distance'})"
        ]
        if self.bandwidth_search_ is not None:
            search_range = self.bandwidth_search_.get("search_range")
            bandwidth_lines.extend(
                [
                    f"Bandwidth criterion: {self.bandwidth_search_.get('criterion')}",
                    f"Bandwidth search method: {self.bandwidth_search_.get('optimization_method')}",
                    f"Bandwidth search range: {search_range}",
                    f"Bandwidth boundary solution: {self.bandwidth_search_.get('boundary_solution')}",
                ]
            )

        rank_deficient_count = (
            int(np.count_nonzero(self.rank_deficient_))
            if self.rank_deficient_ is not None
            else 0
        )

        lines = [
            "=" * 78,
            "Gaussian Geographically Weighted Regression (GWR)",
            "=" * 78,
            f"Samples: {n}",
            f"Predictors: {self.X_train_.shape[1]}",
            f"Kernel: {self.kernel}",
            *bandwidth_lines,
            f"Distance metric: {self.distance_metric}",
            f"Rank-deficient local fits: {rank_deficient_count}/{n}",
            f"Residual variance (sigma^2): {self.sigma2_:.6f}",
            "",
            "Global OLS reference",
            "-" * 78,
            f"{'Variable':<22}{'Estimate':>14}{'Std. Error':>16}",
        ]
        for name, estimate, standard_error in zip(global_names, global_beta, global_se):
            lines.append(f"{name:<22}{estimate:>14.6f}{standard_error:>16.6f}")

        lines.extend(
            [
                "",
                "Local coefficient distribution",
                "-" * 78,
                f"{'Variable':<22}{'Min':>12}{'Median':>12}{'Mean':>12}{'Max':>12}",
            ]
        )
        for index, name in enumerate(global_names):
            values = local_matrix[:, index]
            lines.append(
                f"{name:<22}{np.min(values):>12.6f}{np.median(values):>12.6f}"
                f"{np.mean(values):>12.6f}{np.max(values):>12.6f}"
            )

        lines.extend(["", "GWR diagnostics", "-" * 78])
        ordered = (
            ("R-squared", "r2"),
            ("Adjusted R-squared", "adj_r2"),
            ("RSS", "rss"),
            ("RMSE", "rmse"),
            ("MAE", "mae"),
            ("AIC", "aic"),
            ("AICc", "aicc"),
            ("BIC", "bic"),
            ("trace(S) / ENP v1", "trace_S"),
            ("trace(S'S)", "trace_StS"),
            ("ENP v2", "enp_v2"),
            ("EDF v2", "edf_v2"),
        )
        for label, key in ordered:
            value = self.diagnostics_.get(key, np.nan) if self.diagnostics_ else np.nan
            lines.append(f"{label:<30}{value:>14.6f}")
        lines.append("=" * 78)
        return "\n".join(lines)

    def _reset_fit_state(self) -> None:
        self._mark_unfitted()
        self._reset_regression_state()
        self._reset_gwr_state()
        self._reset_inference_state()
        self.S_matrix_ = None
        self.bandwidth_search_ = None
        self.n_samples_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._reset_robust_state()

    def _weights_from_distances(self, distances: np.ndarray) -> np.ndarray:
        if self.bandwidth_ is None or self.kernel_func_ is None:
            raise RuntimeError("The fitted bandwidth and kernel are unavailable.")
        spatial_weights = _gwr_spatial_weights(
            distances,
            bandwidth=self.bandwidth_,
            adaptive=self.adaptive,
            kernel_func=self.kernel_func_,
        )
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
            self._fit_initial_gwr(
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
        frame = self._gwr_to_frame()
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
        base_summary = self._gwr_summary()
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
