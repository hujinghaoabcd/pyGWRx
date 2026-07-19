# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Scalable geographically weighted regression with polynomial kernels.

The implementation follows the ScaGWR estimator of Murakami et al. (2020).
It pre-compresses Q-nearest-neighbour cross-products before optimizing the
kernel scale and global shrinkage parameters, avoiding an ``n x n`` distance
matrix and repeated local weighted regressions inside calibration.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import t as student_t

from pygwrx.core._summary import format_summary
from pygwrx.core.utils import validate_coords


@dataclass(frozen=True)
class ScalableGWRPredictionResult:
    """Coefficient and prediction results at arbitrary evaluation locations."""

    predictions: Optional[np.ndarray]
    coefficients: np.ndarray
    standard_errors: Optional[np.ndarray]
    coords: np.ndarray
    feature_names: Tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        """Convert the result to a pandas data frame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
        }
        if self.predictions is not None:
            data["prediction"] = self.predictions
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coefficients[:, index]
            if self.standard_errors is not None:
                data[f"se_{name}"] = self.standard_errors[:, index]
        return pd.DataFrame(data)


@dataclass(frozen=True)
class _CompressedMoments:
    """Pre-compressed local cross-products for a set of evaluation sites."""

    cross_products: np.ndarray
    response_products: np.ndarray
    squared_cross_products: Optional[np.ndarray]
    target_X: np.ndarray
    target_y: Optional[np.ndarray]
    coords: np.ndarray


class ScalableGWR:
    """Scalable GWR using a linear multiscale polynomial kernel.

    ScaGWR approximates a continuous Gaussian or exponential kernel by a
    weighted sum of polynomially transformed base kernels. Only ``bandwidth``
    nearest neighbours are used for local cross-products, while ``penalty``
    adds a global OLS cross-product term. The large local cross-products are
    computed once, before optimization, so calibration remains linear in the
    number of observations for fixed feature count, polynomial degree, and
    neighbour count.

    Args:
        bandwidth: Number of nearest neighbours, denoted Q in the paper.
        kernel: Base kernel, either ``"gaussian"`` or ``"exponential"``.
        polynomial: Polynomial degree used to approximate the kernel.
        criterion: Parameter-calibration criterion, ``"cv"`` or ``"aicc"``.
        optimize_bandwidth: Optimize scale and penalty parameters. The neighbour
            count itself is fixed in ScaGWR and is not optimized.
        scale: Fixed positive scale parameter when optimization is disabled, or
            optional initial value when optimization is enabled.
        penalty: Fixed non-negative global shrinkage parameter when optimization
            is disabled, or optional initial value when optimization is enabled.
        fit_intercept: Add a spatially varying intercept.
        sample_size: Optional number of target sites used during CV calibration.
            All observations remain available as neighbours and in the global
            shrinkage term. Ignored for AICc calibration.
        random_state: Random seed used when ``sample_size`` is specified.
        optimizer_maxiter: Maximum L-BFGS-B iterations.
        numerical_jitter: Explicit diagonal stabilization added to every local
            system after the published global penalty term.
        verbose: Print calibration information.

    Notes:
        The historical pyGWRx class with this name was a kNN-truncated ordinary
        GWR that still formed a full distance matrix. It was not ScaGWR. This
        class implements the published polynomial-kernel estimator instead.
    """

    _SUPPORTED_KERNELS = {"gaussian", "exponential"}
    _SUPPORTED_CRITERIA = {"cv", "aicc"}

    def __init__(
        self,
        bandwidth: int = 100,
        kernel: str = "gaussian",
        polynomial: int = 4,
        criterion: str = "cv",
        optimize_bandwidth: bool = True,
        scale: Optional[float] = None,
        penalty: Optional[float] = None,
        fit_intercept: bool = True,
        sample_size: Optional[int] = None,
        random_state: Optional[int] = None,
        optimizer_maxiter: int = 200,
        numerical_jitter: float = 0.0,
        verbose: bool = False,
    ) -> None:
        if not isinstance(bandwidth, (int, np.integer)) or int(bandwidth) < 2:
            raise ValueError("bandwidth must be an integer neighbour count >= 2.")
        kernel_key = str(kernel).strip().lower()
        aliases = {"gau": "gaussian", "exp": "exponential"}
        kernel_key = aliases.get(kernel_key, kernel_key)
        if kernel_key not in self._SUPPORTED_KERNELS:
            raise ValueError(
                "ScalableGWR supports only continuous Gaussian and exponential kernels."
            )
        criterion_key = str(criterion).strip().lower()
        if criterion_key not in self._SUPPORTED_CRITERIA:
            raise ValueError("criterion must be 'cv' or 'aicc'.")
        if not isinstance(polynomial, (int, np.integer)) or int(polynomial) < 1:
            raise ValueError("polynomial must be a positive integer.")
        if scale is not None and (not np.isfinite(scale) or scale <= 0):
            raise ValueError("scale must be finite and positive.")
        if penalty is not None and (not np.isfinite(penalty) or penalty < 0):
            raise ValueError("penalty must be finite and non-negative.")
        if sample_size is not None and (
            not isinstance(sample_size, (int, np.integer)) or int(sample_size) < 2
        ):
            raise ValueError("sample_size must be an integer >= 2.")
        if (
            not isinstance(optimizer_maxiter, (int, np.integer))
            or optimizer_maxiter < 1
        ):
            raise ValueError("optimizer_maxiter must be a positive integer.")
        if not np.isfinite(numerical_jitter) or numerical_jitter < 0:
            raise ValueError("numerical_jitter must be finite and non-negative.")

        self.bandwidth = int(bandwidth)
        self.adaptive = True
        self.kernel = kernel_key
        self.polynomial = int(polynomial)
        self.criterion = criterion_key
        self.optimize_bandwidth = bool(optimize_bandwidth)
        self.scale = scale
        self.penalty = penalty
        self.fit_intercept = bool(fit_intercept)
        self.sample_size = int(sample_size) if sample_size is not None else None
        self.random_state = random_state
        self.optimizer_maxiter = int(optimizer_maxiter)
        self.numerical_jitter = float(numerical_jitter)
        self.verbose = bool(verbose)
        self._clear_fit_state()

    def _clear_fit_state(self) -> None:
        """Clear all fitted attributes before each fit attempt."""
        names = (
            "coefficients_",
            "coef_",
            "intercept_",
            "standard_errors_",
            "coef_standard_errors_",
            "intercept_standard_errors_",
            "t_values_",
            "p_values_",
            "fitted_values_",
            "residuals_",
            "bandwidth_",
            "scale_",
            "penalty_",
            "base_bandwidth_",
            "trace_S_",
            "trace_StS_",
            "effective_n_params_",
            "effective_df_",
            "sigma_",
            "cv_score_",
            "aic_",
            "aicc_",
            "r2_",
            "adjusted_r2_",
            "diagnostics_",
            "optimization_result_",
            "X_train_",
            "y_train_",
            "coords_train_",
            "feature_names_in_",
            "design_feature_names_",
            "global_cross_product_",
            "global_response_product_",
            "_tree_",
        )
        for name in names:
            setattr(self, name, None)
        self.n_features_in_ = None
        self._is_fitted = False

    @staticmethod
    def _as_2d_numeric(
        X: Union[np.ndarray, pd.DataFrame], *, name: str
    ) -> Tuple[np.ndarray, Optional[Tuple[str, ...]]]:
        columns: Optional[Tuple[str, ...]] = None
        if isinstance(X, pd.DataFrame):
            columns = tuple(str(column) for column in X.columns)
            array = X.to_numpy(dtype=float)
        else:
            array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional numeric array.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one row and one column.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return array, columns

    @staticmethod
    def _as_1d_numeric(y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        array = (
            y.to_numpy(dtype=float)
            if isinstance(y, pd.Series)
            else np.asarray(y, dtype=float)
        )
        array = np.ravel(array)
        if array.size == 0:
            raise ValueError("y must contain at least one value.")
        if not np.all(np.isfinite(array)):
            raise ValueError("y contains NaN or infinite values.")
        return array

    def _prepare_design(
        self,
        X: np.ndarray,
        columns: Optional[Tuple[str, ...]],
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if np.any(np.ptp(X, axis=0) == 0):
            bad = np.flatnonzero(np.ptp(X, axis=0) == 0)
            labels = [columns[i] if columns is not None else str(i) for i in bad]
            raise ValueError(
                "ScalableGWR does not accept constant predictor columns; "
                f"constant columns: {labels}."
            )
        names = columns or tuple(f"x{i}" for i in range(X.shape[1]))
        if self.fit_intercept:
            design = np.column_stack((np.ones(X.shape[0]), X))
            design_names = ("Intercept",) + names
        else:
            design = X.copy()
            design_names = names
        return design, design_names

    @staticmethod
    def _drop_self_neighbors(
        indices: np.ndarray,
        distances: np.ndarray,
        target_indices: np.ndarray,
        count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        output_indices = np.empty((len(target_indices), count), dtype=int)
        output_distances = np.empty((len(target_indices), count), dtype=float)
        for row, target in enumerate(target_indices):
            keep = indices[row] != target
            selected_indices = indices[row, keep][:count]
            selected_distances = distances[row, keep][:count]
            if selected_indices.size != count:
                raise ValueError(
                    "Unable to construct the requested leave-one-out neighbourhood."
                )
            output_indices[row] = selected_indices
            output_distances[row] = selected_distances
        return output_indices, output_distances

    def _training_neighbors(
        self,
        coords: np.ndarray,
        target_indices: np.ndarray,
        *,
        exclude_self: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = coords.shape[0]
        q = self.bandwidth
        if exclude_self:
            if q >= n:
                raise ValueError(
                    "bandwidth must be smaller than n_samples for "
                    "leave-one-out calibration."
                )
            query_k = min(n, q + 2)
            distances, indices = self._tree_.query(coords[target_indices], k=query_k)
            if query_k == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            return self._drop_self_neighbors(
                np.asarray(indices), np.asarray(distances), target_indices, q
            )

        query_k = min(n, q)
        distances, indices = self._tree_.query(coords[target_indices], k=query_k)
        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        return np.asarray(indices, dtype=int), np.asarray(distances, dtype=float)

    def _base_kernel(self, distances: np.ndarray) -> np.ndarray:
        h0 = float(self.base_bandwidth_)
        if self.kernel == "gaussian":
            return np.exp(-np.square(distances / h0))
        return np.exp(-distances / h0)

    def _kernel_basis(self, distances: np.ndarray) -> np.ndarray:
        base = self._base_kernel(distances)
        basis = np.ones((*distances.shape, self.polynomial + 1), dtype=float)
        numerator = 2.0 ** (self.polynomial / 2.0)
        for degree in range(1, self.polynomial + 1):
            exponent = numerator / (2.0**degree)
            basis[..., degree] = np.power(base, exponent)
        return basis

    def _compress(
        self,
        target_X: np.ndarray,
        target_coords: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        *,
        target_y: Optional[np.ndarray],
        need_squared: bool,
    ) -> _CompressedMoments:
        X_neighbors = self.X_train_[neighbor_indices]
        y_neighbors = self.y_train_[neighbor_indices]
        basis = self._kernel_basis(neighbor_distances)
        n_targets = target_X.shape[0]
        n_basis = self.polynomial + 1
        n_features = self.X_train_.shape[1]
        cross = np.empty((n_targets, n_basis, n_features, n_features), dtype=float)
        response = np.empty((n_targets, n_basis, n_features), dtype=float)
        squared = np.empty_like(cross) if need_squared else None
        for degree in range(n_basis):
            weights = basis[:, :, degree]
            cross[:, degree] = np.einsum(
                "nqi,nq,nqj->nij", X_neighbors, weights, X_neighbors, optimize=True
            )
            response[:, degree] = np.einsum(
                "nqi,nq,nq->ni", X_neighbors, weights, y_neighbors, optimize=True
            )
            if squared is not None:
                squared[:, degree] = np.einsum(
                    "nqi,nq,nqj->nij",
                    X_neighbors,
                    np.square(weights),
                    X_neighbors,
                    optimize=True,
                )
        return _CompressedMoments(
            cross_products=cross,
            response_products=response,
            squared_cross_products=squared,
            target_X=target_X,
            target_y=target_y,
            coords=target_coords,
        )

    def _basis_coefficients(self, log_scale: float) -> np.ndarray:
        powers = np.arange(1, self.polynomial + 2, dtype=float)
        logits = powers * float(log_scale)
        logits -= np.max(logits)
        coefficients = np.exp(logits)
        return coefficients / np.sum(coefficients)

    def _assemble(
        self,
        moments: _CompressedMoments,
        log_scale: float,
        log_penalty: float,
        *,
        inference: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        coefficients = self._basis_coefficients(log_scale)
        penalty = float(np.exp(log_penalty))
        local_cross = np.einsum(
            "r,nrij->nij", coefficients, moments.cross_products, optimize=True
        )
        local_response = np.einsum(
            "r,nri->ni", coefficients, moments.response_products, optimize=True
        )
        systems = local_cross + penalty * self.global_cross_product_[None, :, :]
        if self.numerical_jitter > 0:
            systems = systems + self.numerical_jitter * np.eye(systems.shape[-1])[None]
        rhs = local_response + penalty * self.global_response_product_[None, :]

        second = None
        if inference:
            if moments.squared_cross_products is None:
                raise RuntimeError("Squared cross-products are required for inference.")
            local_second = np.einsum(
                "r,nrij->nij",
                np.square(coefficients),
                moments.squared_cross_products,
                optimize=True,
            )
            second = (
                local_second
                + 2.0 * penalty * local_cross
                + penalty**2 * self.global_cross_product_[None, :, :]
            )
        return systems, rhs, second

    @staticmethod
    def _solve_batch(systems: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(systems, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "A ScaGWR local system is singular. Increase penalty, bandwidth, "
                "or numerical_jitter."
            ) from exc

    def _evaluate_params(
        self,
        moments: _CompressedMoments,
        log_params: np.ndarray,
        *,
        criterion: str,
    ) -> float:
        try:
            systems, rhs, second = self._assemble(
                moments,
                float(log_params[0]),
                float(log_params[1]),
                inference=criterion == "aicc",
            )
            beta = self._solve_batch(systems, rhs)
            predictions = np.einsum("ni,ni->n", moments.target_X, beta)
            residuals = moments.target_y - predictions
            rss = float(residuals @ residuals)
            if not np.isfinite(rss):
                return 1.0e100
            if criterion == "cv":
                return rss

            inverse_x = np.linalg.solve(systems, moments.target_X[..., None])[..., 0]
            penalty = float(np.exp(log_params[1]))
            trace_s = float(
                (1.0 + penalty) * np.einsum("ni,ni->", moments.target_X, inverse_x)
            )
            n = len(residuals)
            if rss <= 0 or n - 2.0 - trace_s <= 0:
                return 1.0e100
            sigma = np.sqrt(rss / n)
            return float(
                2.0 * n * np.log(sigma)
                + n * np.log(2.0 * np.pi)
                + n * (n + trace_s) / (n - 2.0 - trace_s)
            )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1.0e100

    def _calibrate(self, moments: _CompressedMoments) -> Tuple[float, float]:
        initial_scale = self.scale if self.scale is not None else 1.0
        initial_penalty = self.penalty if self.penalty is not None else 0.01
        if not self.optimize_bandwidth:
            return float(initial_scale), float(initial_penalty)

        initial = np.log([max(initial_scale, 1.0e-12), max(initial_penalty, 1.0e-12)])
        result = minimize(
            lambda params: self._evaluate_params(
                moments, params, criterion=self.criterion
            ),
            initial,
            method="L-BFGS-B",
            bounds=((-12.0, 12.0), (-18.0, 12.0)),
            options={"maxiter": self.optimizer_maxiter, "ftol": 1.0e-10},
        )
        self.optimization_result_ = result
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(
                f"ScaGWR parameter optimization failed: {result.message}"
            )
        if not result.success:
            warnings.warn(
                f"ScaGWR optimizer stopped before convergence: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(np.exp(result.x[0])), float(np.exp(result.x[1]))

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> "ScalableGWR":
        """Fit the published ScaGWR estimator."""
        self._clear_fit_state()
        try:
            X_array, columns = self._as_2d_numeric(X, name="X")
            y_array = self._as_1d_numeric(y)
            coords_array = validate_coords(coords)
            if (
                X_array.shape[0] != y_array.size
                or X_array.shape[0] != coords_array.shape[0]
            ):
                raise ValueError(
                    "X, y, and coords must contain the same number of rows."
                )
            n = X_array.shape[0]
            if self.bandwidth >= n:
                raise ValueError("bandwidth must be smaller than n_samples.")

            design, design_names = self._prepare_design(X_array, columns)
            if self.bandwidth < design.shape[1] + 1:
                raise ValueError(
                    "bandwidth must exceed the number of design columns for "
                    "stable local fitting."
                )
            self.n_features_in_ = X_array.shape[1]
            self.feature_names_in_ = columns or tuple(
                f"x{i}" for i in range(X_array.shape[1])
            )
            self.design_feature_names_ = design_names
            self.X_train_ = design
            self.y_train_ = y_array
            self.coords_train_ = coords_array
            self.bandwidth_ = self.bandwidth
            self.global_cross_product_ = design.T @ design
            self.global_response_product_ = design.T @ y_array
            self._tree_ = cKDTree(coords_array)

            all_indices = np.arange(n, dtype=int)
            nonself_indices, nonself_distances = self._training_neighbors(
                coords_array, all_indices, exclude_self=True
            )
            rank = min(50, self.bandwidth) - 1
            reference_distance = float(np.median(nonself_distances[:, rank]))
            if reference_distance <= 0 or not np.isfinite(reference_distance):
                positive = nonself_distances[nonself_distances > 0]
                if positive.size == 0:
                    raise ValueError(
                        "Coordinates do not provide positive neighbour distances."
                    )
                reference_distance = float(np.median(positive))
            self.base_bandwidth_ = (
                reference_distance / np.sqrt(3.0)
                if self.kernel == "gaussian"
                else reference_distance / 3.0
            )

            if self.criterion == "aicc" and self.sample_size is not None:
                warnings.warn(
                    "sample_size is ignored for AICc calibration.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                calibration_indices = all_indices
            elif self.sample_size is not None and self.sample_size < n:
                rng = np.random.default_rng(self.random_state)
                calibration_indices = np.sort(
                    rng.choice(n, size=self.sample_size, replace=False)
                )
            else:
                calibration_indices = all_indices

            if self.criterion == "cv":
                cv_neighbors, cv_distances = self._training_neighbors(
                    coords_array, calibration_indices, exclude_self=True
                )
                calibration_moments = self._compress(
                    design[calibration_indices],
                    coords_array[calibration_indices],
                    cv_neighbors,
                    cv_distances,
                    target_y=y_array[calibration_indices],
                    need_squared=False,
                )
            else:
                fit_neighbors, fit_distances = self._training_neighbors(
                    coords_array, calibration_indices, exclude_self=False
                )
                calibration_moments = self._compress(
                    design[calibration_indices],
                    coords_array[calibration_indices],
                    fit_neighbors,
                    fit_distances,
                    target_y=y_array[calibration_indices],
                    need_squared=True,
                )

            self.scale_, self.penalty_ = self._calibrate(calibration_moments)
            log_scale = float(np.log(self.scale_))
            log_penalty = float(np.log(max(self.penalty_, 1.0e-300)))

            final_neighbors, final_distances = self._training_neighbors(
                coords_array, all_indices, exclude_self=False
            )
            final_moments = self._compress(
                design,
                coords_array,
                final_neighbors,
                final_distances,
                target_y=y_array,
                need_squared=True,
            )
            systems, rhs, second = self._assemble(
                final_moments, log_scale, log_penalty, inference=True
            )
            beta = self._solve_batch(systems, rhs)
            fitted = np.einsum("ni,ni->n", design, beta)
            residuals = y_array - fitted
            rss = float(residuals @ residuals)

            inverse_x = np.linalg.solve(systems, design[..., None])[..., 0]
            trace_s = float(
                (1.0 + self.penalty_) * np.einsum("ni,ni->", design, inverse_x)
            )
            trace_sts = float(
                np.einsum("ni,nij,nj->", inverse_x, second, inverse_x, optimize=True)
            )
            enp = float(2.0 * trace_s - trace_sts)
            edf = float(n - enp)
            if edf <= 0:
                raise ValueError(
                    "ScaGWR effective residual degrees of freedom are non-positive."
                )
            sigma = float(np.sqrt(rss / edf))

            inverse_systems = np.linalg.inv(systems)
            covariance = np.einsum(
                "nij,njk,nlk->nil",
                inverse_systems,
                second,
                inverse_systems,
                optimize=True,
            )
            standard_errors = sigma * np.sqrt(
                np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                t_values = beta / standard_errors
            p_values = 2.0 * student_t.sf(np.abs(t_values), df=edf)

            full_cv_neighbors, full_cv_distances = self._training_neighbors(
                coords_array, all_indices, exclude_self=True
            )
            full_cv_moments = self._compress(
                design,
                coords_array,
                full_cv_neighbors,
                full_cv_distances,
                target_y=y_array,
                need_squared=False,
            )
            cv_rss = self._evaluate_params(
                full_cv_moments,
                np.array([log_scale, log_penalty]),
                criterion="cv",
            )
            self.cv_score_ = float(np.sqrt(cv_rss / n))

            sigma_ml = np.sqrt(max(rss / n, np.finfo(float).tiny))
            log_likelihood = float(
                -n * np.log(sigma_ml) - n / 2.0 * np.log(2.0 * np.pi)
            )
            aic = float(-2.0 * log_likelihood + n + trace_s)
            aicc = (
                float(-2.0 * log_likelihood + n * (n + trace_s) / (n - 2.0 - trace_s))
                if n - 2.0 - trace_s > 0
                else np.inf
            )
            tss = float(np.sum(np.square(y_array - np.mean(y_array))))
            rss_r2 = 1.0 - rss / tss if tss > 0 else np.nan
            corr = np.corrcoef(y_array, fitted)[0, 1]
            r2 = float(corr * corr) if np.isfinite(corr) else float(rss_r2)
            adjusted_r2 = float(1.0 - (1.0 - r2) * (n - 1.0) / (n - enp - 1.0))

            self.coefficients_ = beta
            if self.fit_intercept:
                self.intercept_ = beta[:, 0]
                self.coef_ = beta[:, 1:]
                self.intercept_standard_errors_ = standard_errors[:, 0]
                self.coef_standard_errors_ = standard_errors[:, 1:]
            else:
                self.intercept_ = np.zeros(n)
                self.coef_ = beta
                self.intercept_standard_errors_ = np.zeros(n)
                self.coef_standard_errors_ = standard_errors
            self.standard_errors_ = standard_errors
            self.t_values_ = t_values
            self.p_values_ = p_values
            self.fitted_values_ = fitted
            self.residuals_ = residuals
            self.trace_S_ = trace_s
            self.trace_StS_ = trace_sts
            self.effective_n_params_ = enp
            self.effective_df_ = edf
            self.sigma_ = sigma
            self.aic_ = aic
            self.aicc_ = aicc
            self.r2_ = r2
            self.adjusted_r2_ = adjusted_r2
            self.diagnostics_ = {
                "rss": rss,
                "sigma": sigma,
                "trace_S": trace_s,
                "trace_StS": trace_sts,
                "enp": enp,
                "edf": edf,
                "r2": r2,
                "rss_r2": float(rss_r2),
                "adjusted_r2": adjusted_r2,
                "aic": aic,
                "aicc": aicc,
                "cv_rmse": self.cv_score_,
                "scale": self.scale_,
                "penalty": self.penalty_,
                "base_bandwidth": self.base_bandwidth_,
                "n_neighbors": self.bandwidth_,
            }
            self._is_fitted = True

            if self.verbose:
                print(
                    "ScalableGWR fitted: "
                    f"n={n}, Q={self.bandwidth_}, scale={self.scale_:.6g}, "
                    f"penalty={self.penalty_:.6g}, CV_RMSE={self.cv_score_:.6g}"
                )
            return self
        except Exception:
            self._clear_fit_state()
            raise

    def _check_prediction_features(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        array, columns = self._as_2d_numeric(X, name="X")
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {array.shape[1]} features; expected {self.n_features_in_}."
            )
        if isinstance(X, pd.DataFrame) and columns != self.feature_names_in_:
            raise ValueError(
                "Prediction DataFrame columns must match the fitted columns "
                "in the same order."
            )
        if self.fit_intercept:
            return np.column_stack((np.ones(array.shape[0]), array))
        return array

    def predict_result(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]],
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        return_standard_errors: bool = False,
    ) -> ScalableGWRPredictionResult:
        """Estimate coefficients and optionally predictions at new locations."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        coords_array = validate_coords(coords)
        if X is None:
            target_design = np.zeros((coords_array.shape[0], self.X_train_.shape[1]))
            predictions = None
        else:
            target_design = self._check_prediction_features(X)
            if target_design.shape[0] != coords_array.shape[0]:
                raise ValueError("X and coords must contain the same number of rows.")
            predictions = np.empty(coords_array.shape[0], dtype=float)

        distances, indices = self._tree_.query(coords_array, k=self.bandwidth_)
        if self.bandwidth_ == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        moments = self._compress(
            target_design,
            coords_array,
            np.asarray(indices, dtype=int),
            np.asarray(distances, dtype=float),
            target_y=None,
            need_squared=return_standard_errors,
        )
        systems, rhs, second = self._assemble(
            moments,
            float(np.log(self.scale_)),
            float(np.log(max(self.penalty_, 1.0e-300))),
            inference=return_standard_errors,
        )
        beta = self._solve_batch(systems, rhs)
        if predictions is not None:
            predictions[:] = np.einsum("ni,ni->n", target_design, beta)

        standard_errors = None
        if return_standard_errors:
            inverse_systems = np.linalg.inv(systems)
            covariance = np.einsum(
                "nij,njk,nlk->nil",
                inverse_systems,
                second,
                inverse_systems,
                optimize=True,
            )
            standard_errors = self.sigma_ * np.sqrt(
                np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0)
            )
        return ScalableGWRPredictionResult(
            predictions=predictions,
            coefficients=beta,
            standard_errors=standard_errors,
            coords=coords_array,
            feature_names=self.design_feature_names_,
        )

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Predict responses by estimating ScaGWR coefficients at new locations."""
        result = self.predict_result(X, coords)
        return np.asarray(result.predictions)

    def to_frame(self) -> pd.DataFrame:
        """Return training-location coefficients, inference, and fit diagnostics."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords_train_[:, 0],
            "coord_1": self.coords_train_[:, 1],
            "observed": self.y_train_,
            "fitted": self.fitted_values_,
            "residual": self.residuals_,
        }
        for index, name in enumerate(self.design_feature_names_):
            data[f"coef_{name}"] = self.coefficients_[:, index]
            data[f"se_{name}"] = self.standard_errors_[:, index]
            data[f"t_{name}"] = self.t_values_[:, index]
            data[f"p_{name}"] = self.p_values_[:, index]
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Return fitted diagnostics as a plain-text table."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return format_summary("Scalable GWR Summary", self.diagnostics_)
