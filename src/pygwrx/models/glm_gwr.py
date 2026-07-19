# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Geographically weighted generalized linear models.

This module implements Gaussian, Poisson, and Bernoulli geographically
weighted regression with local iteratively weighted least squares, model-
appropriate diagnostics, exposure-aware Poisson prediction, and automatic
bandwidth selection.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln

from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.optimization import (
    BrentSearch,
    GoldenSectionSearch,
    OptimizationResult,
)
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import add_intercept, compute_distance_matrix, validate_coords
from pygwrx.models.gwr import GWR, GWRPredictionResult

FamilyName = Literal["gaussian", "poisson", "binomial"]
KernelLike = Union[str, Callable[[np.ndarray, float], np.ndarray]]
BandwidthLike = Union[float, int, str, None]
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]

_EPS = np.finfo(float).eps
_RIDGE = 1.0e-8


@dataclass(frozen=True)
class GWGLMPredictionResult:
    """Rich prediction result returned by :meth:`GWGLM.predict_result`."""

    predictions: np.ndarray
    linear_predictor: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    coords: np.ndarray
    feature_names: Tuple[str, ...]
    family: str
    exposure: Optional[np.ndarray] = None
    coef_standard_errors: Optional[np.ndarray] = None
    intercept_standard_errors: Optional[np.ndarray] = None
    coef_z_values: Optional[np.ndarray] = None
    intercept_z_values: Optional[np.ndarray] = None

    def to_frame(self) -> pd.DataFrame:
        """Return prediction results as a pandas DataFrame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "prediction": self.predictions,
            "linear_predictor": self.linear_predictor,
            "intercept": self.intercept,
        }
        if self.exposure is not None:
            data["exposure"] = self.exposure
        if self.intercept_standard_errors is not None:
            data["intercept_se"] = self.intercept_standard_errors
        if self.intercept_z_values is not None:
            data["intercept_z"] = self.intercept_z_values
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coef[:, index]
            if self.coef_standard_errors is not None:
                data[f"se_{name}"] = self.coef_standard_errors[:, index]
            if self.coef_z_values is not None:
                data[f"z_{name}"] = self.coef_z_values[:, index]
        return pd.DataFrame(data)

    def to_geodataframe(self, crs: Optional[Union[str, int]] = None):
        """Return prediction results as a point GeoDataFrame."""
        from pygwrx.io import to_geodataframe

        frame = self.to_frame()
        columns = [column for column in frame if not column.startswith("coord_")]
        return to_geodataframe(
            frame[columns].to_numpy(dtype=float),
            None,
            self.coords,
            feature_names=columns,
            crs=crs,
        )


@dataclass
class _LocalIWLSResult:
    params: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    sqrt_glm_weights: np.ndarray
    inverse_xtx_xt: np.ndarray
    n_iter: int
    converged: bool


@dataclass
class _GWGLMFitResult:
    params: np.ndarray
    fitted_values: np.ndarray
    linear_predictor: np.ndarray
    influence: np.ndarray
    trace_S: float
    trace_StS: float
    covariance_factors: np.ndarray
    iteration_counts: np.ndarray
    converged: np.ndarray
    final_working_weights: np.ndarray
    hat_matrix: Optional[np.ndarray]


class GWGLM(GWR):
    """Geographically weighted generalized linear model.

    The estimator supports three canonical families:

    * ``"gaussian"`` with an identity link;
    * ``"poisson"`` with a log link and optional exposure or log-offset;
    * ``"binomial"`` for Bernoulli responses with a logit link.

    Poisson and Binomial models are fitted independently at each regression
    location by local iteratively weighted least squares (IWLS). Spatial kernel
    weights and GLM working weights are multiplied inside each local fit.

    Args:
        family: Response distribution. Supported values are ``"gaussian"``,
            ``"poisson"``, and ``"binomial"``.
        kernel: Spatial kernel name or callable.
        bandwidth: Numeric bandwidth or automatic-selection criterion. A fixed
            bandwidth is a distance; an adaptive bandwidth is a neighbour count.
        bandwidth_method: Selection criterion used when ``bandwidth=None``.
        adaptive: Whether the bandwidth represents nearest-neighbour count.
        bandwidth_range: Optional lower and upper search bounds.
        optimization_method: ``"golden_section"``, ``"brent"``, or ``"grid"``.
        max_iter: Maximum IWLS iterations per local model.
        tol: IWLS convergence tolerance.
        fit_intercept: Whether to include a local intercept.
        distance_metric: Distance metric used by the spatial kernel.
        sigma2_v1: Gaussian residual-variance convention inherited from GWR.
        verbose: Whether to print fitting and bandwidth-search progress.

    Attributes:
        family_: Normalized fitted family name.
        bandwidth_: Selected or user-specified bandwidth.
        coef_: Local slope coefficients.
        intercept_: Local intercepts.
        fitted_values_: Fitted conditional means.
        linear_predictor_: Fitted values on the link scale.
        residuals_: Response residuals ``y - fitted_values_``.
        deviance_residuals_: Signed deviance residuals.
        pearson_residuals_: Pearson residuals.
        parameter_standard_errors_: Local parameter standard errors.
        parameter_z_values_: Local Wald z statistics for non-Gaussian models.
        iteration_counts_: Number of IWLS iterations at each location.
        converged_: Whether every local IWLS fit converged.
        exposure_train_: Poisson exposure used during fitting.

    Notes:
        ``offset`` in :meth:`fit` and :meth:`predict` is an additive offset on
        the linear-predictor scale. For Poisson models it is equivalent to
        ``log(exposure)``. Supply at most one of ``exposure`` and ``offset``.

        Grouped-binomial responses are intentionally not accepted. The current
        Binomial implementation is Bernoulli only and requires values in
        ``{0, 1}``.

    References:
        Nakaya, T., Fotheringham, A. S., Brunsdon, C., and Charlton, M. (2005).
        Geographically weighted Poisson regression for disease association
        mapping. Statistics in Medicine, 24, 2695-2717.

        Oshan, T. M., Li, Z., Kang, W., Wolf, L. J., and Fotheringham, A. S.
        (2019). mgwr: A Python implementation of multiscale geographically
        weighted regression for investigating process spatial heterogeneity and
        scale. ISPRS International Journal of Geo-Information, 8, 269.
    """

    def __init__(
        self,
        family: FamilyName = "gaussian",
        kernel: KernelLike = "bisquare",
        bandwidth: BandwidthLike = "cv",
        bandwidth_method: str = "aicc",
        adaptive: bool = False,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        optimization_method: str = "golden_section",
        max_iter: int = 100,
        tol: float = 1.0e-6,
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        verbose: bool = False,
    ) -> None:
        normalized_family = self._normalize_family(family)
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
            max_iter, (int, np.integer)
        ):
            raise TypeError("max_iter must be a positive integer.")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be greater than zero.")
        if isinstance(tol, (bool, np.bool_)):
            raise TypeError("tol must be a positive real scalar.")
        tol_value = float(tol)
        if not np.isfinite(tol_value) or tol_value <= 0.0:
            raise ValueError("tol must be finite and greater than zero.")

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
        self.family = normalized_family
        self.max_iter = int(max_iter)
        self.tol = tol_value
        self._reset_glm_state()

    @staticmethod
    def _normalize_family(family: object) -> FamilyName:
        if not isinstance(family, str) or not family.strip():
            raise ValueError(
                "family must be one of 'gaussian', 'poisson', or 'binomial'."
            )
        value = family.strip().lower()
        if value == "bernoulli":
            value = "binomial"
        if value not in {"gaussian", "poisson", "binomial"}:
            raise ValueError(
                "family must be one of 'gaussian', 'poisson', or 'binomial'. "
                "Gamma is not exposed because it has not been validated against a "
                "mature geographically weighted reference implementation."
            )
        return value  # type: ignore[return-value]

    def _reset_glm_state(self) -> None:
        self.family_: Optional[str] = None
        self.exposure_train_: Optional[np.ndarray] = None
        self.offset_train_: Optional[np.ndarray] = None
        self.linear_predictor_: Optional[np.ndarray] = None
        self.mu_: Optional[np.ndarray] = None
        self.deviance_: Optional[float] = None
        self.null_deviance_: Optional[float] = None
        self.percent_deviance_: Optional[float] = None
        self.adjusted_percent_deviance_: Optional[float] = None
        self.log_likelihood_: Optional[float] = None
        self.deviance_residuals_: Optional[np.ndarray] = None
        self.pearson_residuals_: Optional[np.ndarray] = None
        self.iteration_counts_: Optional[np.ndarray] = None
        self.local_converged_: Optional[np.ndarray] = None
        self.converged_: Optional[bool] = None
        self.final_working_weights_: Optional[np.ndarray] = None
        self.parameter_z_values_: Optional[np.ndarray] = None
        self.intercept_z_: Optional[np.ndarray] = None
        self.coef_z_: Optional[np.ndarray] = None
        self.bandwidth_selection_result_: Optional[OptimizationResult] = None
        self.bandwidth_selection_score_: Optional[float] = None
        self.cv_residuals_: Optional[np.ndarray] = None
        self.cv_contributions_: Optional[np.ndarray] = None

    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self._reset_glm_state()

    @staticmethod
    def _as_vector(values: object, n_samples: int, name: str) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values.") from exc
        if array.ndim == 0:
            array = np.full(n_samples, float(array), dtype=float)
        elif array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        elif array.ndim != 1:
            raise ValueError(f"{name} must be scalar or one-dimensional.")
        if array.shape[0] != n_samples:
            raise ValueError(
                f"{name} must contain {n_samples} values; received {array.shape[0]}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return np.asarray(array, dtype=float)

    def _prepare_exposure(
        self,
        n_samples: int,
        *,
        exposure: Optional[object],
        offset: Optional[object],
    ) -> tuple[np.ndarray, np.ndarray]:
        if exposure is not None and offset is not None:
            raise ValueError("Provide either exposure or offset, not both.")
        if self.family != "poisson":
            if exposure is not None or offset is not None:
                raise ValueError(
                    "exposure and offset are supported only for Poisson GWGLM."
                )
            ones = np.ones(n_samples, dtype=float)
            zeros = np.zeros(n_samples, dtype=float)
            return ones, zeros
        if offset is not None:
            offset_arr = self._as_vector(offset, n_samples, "offset")
            exposure_arr = np.exp(offset_arr)
            if not np.all(np.isfinite(exposure_arr)) or np.any(exposure_arr <= 0.0):
                raise ValueError("exp(offset) must be finite and strictly positive.")
            return exposure_arr, offset_arr
        if exposure is None:
            exposure_arr = np.ones(n_samples, dtype=float)
        else:
            exposure_arr = self._as_vector(exposure, n_samples, "exposure")
        if np.any(exposure_arr <= 0.0):
            raise ValueError("exposure must be strictly positive.")
        return exposure_arr, np.log(exposure_arr)

    def _validate_response(self, y: np.ndarray) -> None:
        if self.family == "poisson" and np.any(y < 0.0):
            raise ValueError("Poisson response values must be non-negative.")
        if self.family == "binomial":
            unique = np.unique(y)
            if not np.all(np.isin(unique, [0.0, 1.0])):
                raise ValueError(
                    "Binomial GWGLM currently supports Bernoulli responses only; "
                    "y must contain only 0 and 1."
                )
            if unique.size < 2:
                raise ValueError("Binomial response must contain both outcome classes.")

    @staticmethod
    def _solve(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(system) @ rhs
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                "The local IWLS system produced non-finite values."
            )
        return solution

    def _iwls(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        spatial_weights: np.ndarray,
        exposure: np.ndarray,
        *,
        initial_params: Optional[np.ndarray] = None,
    ) -> _LocalIWLSResult:
        n_samples, n_params = X_design.shape
        if initial_params is None:
            beta = np.zeros(n_params, dtype=float)
        else:
            beta = np.asarray(initial_params, dtype=float).reshape(-1).copy()
            if beta.size != n_params:
                raise ValueError(
                    f"initial_params must contain {n_params} values; received {beta.size}."
                )

        if initial_params is not None:
            eta = X_design @ beta
            if self.family == "poisson":
                mu = np.exp(np.clip(eta, -700.0, 700.0)) * exposure
            elif self.family == "binomial":
                mu = expit(eta)
            else:
                raise RuntimeError(
                    "Local IWLS is used only for Poisson and Binomial families."
                )
        elif self.family == "poisson":
            response_rate = y / exposure
            initial_rate = 0.5 * (response_rate + np.mean(response_rate))
            eta = np.log(np.clip(initial_rate, _EPS, None))
            mu = 0.5 * (y + np.mean(y))
            mu = np.clip(mu, _EPS, None)
        elif self.family == "binomial":
            mu = np.clip((y + 0.5) / 2.0, _EPS, 1.0 - _EPS)
            eta = np.log(mu / (1.0 - mu))
        else:
            raise RuntimeError(
                "Local IWLS is used only for Poisson and Binomial families."
            )

        converged = False
        inverse_xtx_xt = np.empty((n_params, n_samples), dtype=float)
        sqrt_glm_weights = np.ones(n_samples, dtype=float)
        for iteration in range(1, self.max_iter + 1):
            if self.family == "poisson":
                mu_safe = np.clip(mu, _EPS, None)
                glm_variance = mu_safe
                z = eta + (y - mu_safe) / mu_safe
            else:
                mu_safe = np.clip(mu, _EPS, 1.0 - _EPS)
                glm_variance = mu_safe * (1.0 - mu_safe)
                z = eta + (y - mu_safe) / glm_variance

            sqrt_glm_weights = np.sqrt(np.clip(glm_variance, _EPS, None))
            weighted_X = X_design * sqrt_glm_weights[:, None]
            weighted_z = z * sqrt_glm_weights
            XtW = weighted_X.T * spatial_weights
            system = XtW @ weighted_X + _RIDGE * np.eye(n_params, dtype=float)
            inverse = self._solve(system, np.eye(n_params, dtype=float))
            inverse_xtx_xt = inverse @ XtW
            new_beta = inverse_xtx_xt @ weighted_z

            eta = X_design @ new_beta
            if self.family == "poisson":
                mu = np.exp(np.clip(eta, -700.0, 700.0)) * exposure
            else:
                mu = expit(eta)

            # mgwr/spglm uses the smallest absolute coefficient update as its
            # historical stopping rule. Matching it preserves numerical parity
            # with the established Python reference implementation.
            difference = float(np.min(np.abs(new_beta - beta)))
            beta = np.asarray(new_beta, dtype=float)
            if difference <= self.tol:
                converged = True
                break

        return _LocalIWLSResult(
            params=beta,
            mu=np.asarray(mu, dtype=float),
            eta=np.asarray(eta, dtype=float),
            sqrt_glm_weights=np.asarray(sqrt_glm_weights, dtype=float),
            inverse_xtx_xt=np.asarray(inverse_xtx_xt, dtype=float),
            n_iter=iteration,
            converged=converged,
        )

    def _candidate_weights(
        self,
        distance_row: np.ndarray,
        bandwidth: Union[int, float],
    ) -> np.ndarray:
        local_bandwidth = (
            adaptive_bandwidth_weights(distance_row, int(bandwidth))
            if self.adaptive
            else float(bandwidth)
        )
        weights = np.asarray(
            self.kernel_func_(distance_row, local_bandwidth), dtype=float
        )
        if weights.shape != distance_row.shape:
            raise ValueError("The kernel returned an unexpected weight shape.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("The kernel returned invalid spatial weights.")
        if not np.any(weights > 0.0):
            raise ValueError("The local kernel contains no positive weights.")
        return weights

    def _fit_non_gaussian_locations(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        exposure: np.ndarray,
        bandwidth: Union[int, float],
        *,
        store_hat_matrix: bool,
    ) -> _GWGLMFitResult:
        distances = compute_distance_matrix(coords, coords, metric=self.distance_metric)
        n_samples, n_params = X_design.shape
        params = np.empty((n_samples, n_params), dtype=float)
        fitted = np.empty(n_samples, dtype=float)
        eta_fitted = np.empty(n_samples, dtype=float)
        influence = np.empty(n_samples, dtype=float)
        covariance_factors = np.empty((n_samples, n_params), dtype=float)
        iteration_counts = np.empty(n_samples, dtype=int)
        local_converged = np.empty(n_samples, dtype=bool)
        final_weights = np.empty(n_samples, dtype=float)
        hat_matrix = (
            np.empty((n_samples, n_samples), dtype=float) if store_hat_matrix else None
        )
        trace_s = 0.0
        trace_sts = 0.0

        for index, distance_row in enumerate(distances):
            spatial_weights = self._candidate_weights(distance_row, bandwidth)
            local = self._iwls(X_design, y, spatial_weights, exposure)
            raw_hat_row = X_design[index] @ local.inverse_xtx_xt
            hat_row = raw_hat_row * local.sqrt_glm_weights
            params[index] = local.params
            fitted[index] = local.mu[index]
            eta_fitted[index] = local.eta[index]
            influence[index] = hat_row[index]
            trace_s += float(hat_row[index])
            trace_sts += float(np.dot(hat_row, hat_row))
            covariance_factors[index] = np.sum(
                local.inverse_xtx_xt * local.inverse_xtx_xt, axis=1
            )
            iteration_counts[index] = local.n_iter
            local_converged[index] = local.converged
            final_weights[index] = local.sqrt_glm_weights[index]
            if hat_matrix is not None:
                hat_matrix[index] = hat_row

        return _GWGLMFitResult(
            params=params,
            fitted_values=fitted,
            linear_predictor=eta_fitted,
            influence=influence,
            trace_S=trace_s,
            trace_StS=trace_sts,
            covariance_factors=covariance_factors,
            iteration_counts=iteration_counts,
            converged=local_converged,
            final_working_weights=final_weights,
            hat_matrix=hat_matrix,
        )

    def _deviance_residuals(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        if self.family == "poisson":
            mu_safe = np.clip(mu, _EPS, None)
            terms = np.where(
                y > 0.0,
                y * np.log(np.clip(y / mu_safe, _EPS, None)) - (y - mu_safe),
                mu_safe,
            )
            return np.sign(y - mu_safe) * np.sqrt(np.maximum(2.0 * terms, 0.0))
        if self.family == "binomial":
            mu_safe = np.clip(mu, _EPS, 1.0 - _EPS)
            likelihood = np.where(y == 1.0, mu_safe, 1.0 - mu_safe)
            return np.sign(y - mu_safe) * np.sqrt(-2.0 * np.log(likelihood))
        return y - mu

    def _deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        residuals = self._deviance_residuals(y, mu)
        return float(np.dot(residuals, residuals))

    def _log_likelihood(self, y: np.ndarray, mu: np.ndarray) -> float:
        if self.family == "poisson":
            mu_safe = np.clip(mu, _EPS, None)
            return float(np.sum(y * np.log(mu_safe) - mu_safe - gammaln(y + 1.0)))
        if self.family == "binomial":
            mu_safe = np.clip(mu, _EPS, 1.0 - _EPS)
            return float(
                np.sum(y * np.log(mu_safe) + (1.0 - y) * np.log(1.0 - mu_safe))
            )
        residual = y - mu
        rss = max(float(np.dot(residual, residual)), _EPS)
        n = y.size
        return float(-0.5 * n * (np.log(2.0 * np.pi * rss / n) + 1.0))

    def _null_mean(self, y: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        if self.family == "poisson":
            rate = float(np.sum(y) / np.sum(exposure))
            return exposure * max(rate, _EPS)
        if self.family == "binomial":
            probability = float(np.clip(np.mean(y), _EPS, 1.0 - _EPS))
            return np.full_like(y, probability, dtype=float)
        return np.full_like(y, np.mean(y), dtype=float)

    def _information_diagnostics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        *,
        trace_s: float,
        trace_sts: float,
    ) -> Dict[str, float]:
        n = y.size
        deviance = self._deviance(y, mu)
        null_deviance = self._deviance(y, self._null_mean(y, exposure))
        log_likelihood = self._log_likelihood(y, mu)
        effective_params = float(trace_s)
        denominator = n - effective_params - 1.0
        aic = deviance + 2.0 * effective_params
        aicc = (
            aic + 2.0 * effective_params * (effective_params + 1.0) / denominator
            if denominator > 0.0
            else np.inf
        )
        bic = deviance + effective_params * np.log(n)
        d2 = 1.0 - deviance / null_deviance if null_deviance > _EPS else np.nan
        adj_d2 = (
            1.0 - (1.0 - d2) * (n - 1.0) / denominator
            if denominator > 0.0 and np.isfinite(d2)
            else np.nan
        )
        return {
            "deviance": float(deviance),
            "null_deviance": float(null_deviance),
            "log_likelihood": float(log_likelihood),
            "percent_deviance": float(d2),
            "adjusted_percent_deviance": float(adj_d2),
            "trace_S": float(trace_s),
            "trace_StS": float(trace_sts),
            "effective_params": effective_params,
            "edf": float(n - 2.0 * trace_s + trace_sts),
            "aic": float(aic),
            "aicc": float(aicc),
            "bic": float(bic),
        }

    def _default_bandwidth_range(
        self, distances: np.ndarray, n_params: int
    ) -> tuple[Union[int, float], Union[int, float]]:
        n_samples = distances.shape[0]
        if self.bandwidth_range is not None:
            lower, upper = self.bandwidth_range
            if self.adaptive:
                lower_int = max(int(np.ceil(lower)), n_params + 1)
                upper_int = min(int(np.floor(upper)), n_samples)
                if lower_int > upper_int:
                    raise ValueError(
                        "bandwidth_range contains no valid adaptive candidate."
                    )
                return lower_int, upper_int
            return float(lower), float(upper)
        if self.adaptive:
            return max(n_params + 1, int(np.ceil(0.05 * n_samples)), 2), n_samples
        upper = distances[np.triu_indices_from(distances, k=1)]
        positive = upper[np.isfinite(upper) & (upper > 0.0)]
        if positive.size == 0:
            raise ValueError(
                "Automatic fixed bandwidth selection requires distinct coordinates."
            )
        lower_value = float(np.percentile(positive, 5.0))
        upper_value = float(np.percentile(positive, 95.0))
        if lower_value >= upper_value:
            lower_value = float(np.min(positive))
            upper_value = float(np.max(positive))
        if lower_value >= upper_value:
            upper_value = float(np.nextafter(lower_value, np.inf))
        return lower_value, upper_value

    def _leave_one_out_score(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        exposure: np.ndarray,
        bandwidth: Union[int, float],
        *,
        return_residuals: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        distances = compute_distance_matrix(coords, coords, metric=self.distance_metric)
        residuals = np.empty(y.size, dtype=float)
        for index, distance_row in enumerate(distances):
            spatial_weights = self._candidate_weights(distance_row, bandwidth)
            spatial_weights[index] = 0.0
            if np.count_nonzero(spatial_weights > 0.0) < X_design.shape[1]:
                residuals[index] = np.inf
                continue
            local = self._iwls(X_design, y, spatial_weights, exposure)
            eta = float(X_design[index] @ local.params)
            if self.family == "poisson":
                prediction = float(
                    np.exp(np.clip(eta, -700.0, 700.0)) * exposure[index]
                )
            else:
                prediction = float(expit(eta))
            residuals[index] = y[index] - prediction
        score = (
            float(np.dot(residuals, residuals))
            if np.all(np.isfinite(residuals))
            else np.inf
        )
        return (score, residuals) if return_residuals else score

    def _selection_objective(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        exposure: np.ndarray,
        method: str,
        bandwidth: Union[int, float],
    ) -> float:
        try:
            if method == "cv":
                return float(
                    self._leave_one_out_score(X_design, y, coords, exposure, bandwidth)
                )
            result = self._fit_non_gaussian_locations(
                X_design,
                y,
                coords,
                exposure,
                bandwidth,
                store_hat_matrix=False,
            )
            diagnostics = self._information_diagnostics(
                y,
                result.fitted_values,
                exposure,
                trace_s=result.trace_S,
                trace_sts=result.trace_StS,
            )
            return float(diagnostics[method])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return np.inf

    def _resolve_non_gaussian_bandwidth(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        exposure: np.ndarray,
    ) -> Union[int, float]:
        if not isinstance(self.bandwidth, str) and self.bandwidth is not None:
            value = float(self.bandwidth)
            if self.adaptive:
                if not value.is_integer():
                    raise ValueError(
                        "adaptive bandwidth must be an integer neighbour count."
                    )
                candidate = int(value)
                if candidate < X_design.shape[1] + 1 or candidate > y.size:
                    raise ValueError(
                        "adaptive bandwidth must be at least n_parameters + 1 and no "
                        "larger than n_samples."
                    )
                return candidate
            return value

        method = (
            self.bandwidth.strip().lower()
            if isinstance(self.bandwidth, str)
            else self.bandwidth_method.strip().lower()
        )
        if method not in {"cv", "aic", "aicc", "bic"}:
            raise ValueError(
                "GWGLM bandwidth method must be 'cv', 'aic', 'aicc', or 'bic'."
            )
        distances = compute_distance_matrix(coords, coords, metric=self.distance_metric)
        lower, upper = self._default_bandwidth_range(distances, X_design.shape[1])

        def objective(candidate: float) -> float:
            normalized: Union[int, float] = (
                int(round(candidate)) if self.adaptive else float(candidate)
            )
            score = self._selection_objective(
                X_design, y, coords, exposure, method, normalized
            )
            if self.verbose:
                print(f"  bandwidth={normalized}: {method}={score:.8g}")
            return score

        if self.optimization_method == "grid":
            if self.adaptive:
                candidates = np.arange(int(lower), int(upper) + 1, dtype=int)
            else:
                candidates = np.linspace(float(lower), float(upper), 40)
            scores = np.asarray(
                [objective(float(candidate)) for candidate in candidates]
            )
            best_index = int(np.argmin(scores))
            best_value: Union[int, float] = (
                int(candidates[best_index])
                if self.adaptive
                else float(candidates[best_index])
            )
            result = OptimizationResult(
                value=best_value,
                score=float(scores[best_index]),
                iterations=0,
                converged=bool(np.isfinite(scores[best_index])),
                evaluations=int(candidates.size),
                message="Exhaustive/grid bandwidth search completed.",
            )
        elif self.optimization_method == "brent" and not self.adaptive:
            result = BrentSearch(
                tol=1.0e-5, max_iter=100, verbose=self.verbose
            ).minimize(objective, float(lower), float(upper))
        else:
            result = GoldenSectionSearch(
                tol=1.0e-5,
                max_iter=100,
                verbose=self.verbose,
            ).minimize(objective, float(lower), float(upper), adaptive=self.adaptive)
        if not np.isfinite(result.score):
            raise RuntimeError("No estimable GWGLM bandwidth candidate was found.")
        self.bandwidth_selection_result_ = result
        self.bandwidth_selection_score_ = float(result.score)
        return int(result.value) if self.adaptive else float(result.value)

    def _set_non_gaussian_inference(
        self,
        fit_result: _GWGLMFitResult,
        diagnostics: Dict[str, float],
    ) -> None:
        self.influence_ = fit_result.influence.copy()
        standard_errors = np.sqrt(np.maximum(fit_result.covariance_factors, 0.0))
        full_params = (
            np.column_stack([self.intercept_, self.coef_])
            if self.fit_intercept
            else np.asarray(self.coef_)
        )
        z_values = np.full_like(full_params, np.nan, dtype=float)
        np.divide(
            full_params,
            standard_errors,
            out=z_values,
            where=standard_errors > _EPS,
        )
        self.parameter_covariance_diagonal_ = fit_result.covariance_factors.copy()
        self.parameter_standard_errors_ = standard_errors
        self.parameter_t_values_ = (
            z_values.copy()
        )  # compatibility with GWR result schema
        self.parameter_z_values_ = z_values
        if self.fit_intercept:
            self.intercept_se_ = standard_errors[:, 0]
            self.coef_se_ = standard_errors[:, 1:]
            self.intercept_t_ = z_values[:, 0]
            self.coef_t_ = z_values[:, 1:]
            self.intercept_z_ = z_values[:, 0]
            self.coef_z_ = z_values[:, 1:]
        else:
            self.intercept_se_ = np.zeros(self.n_samples_, dtype=float)
            self.coef_se_ = standard_errors
            self.intercept_t_ = np.full(self.n_samples_, np.nan, dtype=float)
            self.coef_t_ = z_values
            self.intercept_z_ = self.intercept_t_.copy()
            self.coef_z_ = z_values
        self.sigma2_ = 1.0
        residual_variance = np.maximum(1.0 - self.influence_, _EPS)
        self.standardized_residuals_ = self.residuals_ / np.sqrt(residual_variance)
        effective_params = max(float(diagnostics["effective_params"]), _EPS)
        self.cooks_distance_ = (
            self.standardized_residuals_**2
            * self.influence_
            / (effective_params * np.maximum(1.0 - self.influence_, _EPS))
        )
        self.inference_enabled_ = True

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        coords: ArrayLike,
        *,
        exposure: Optional[object] = None,
        offset: Optional[object] = None,
        compute_hat_matrix: bool = False,
        compute_inference: bool = True,
        compute_local_r2: bool = True,
    ) -> "GWGLM":
        """Fit the geographically weighted generalized linear model.

        Args:
            X: Predictor matrix with shape ``(n_samples, n_features)``.
            y: Response vector. Poisson values must be non-negative; Binomial
                values must be Bernoulli outcomes in ``{0, 1}``.
            coords: Spatial coordinates with shape ``(n_samples, 2)``.
            exposure: Positive Poisson exposure. Scalar values are broadcast.
            offset: Poisson log-exposure offset. Supply at most one of exposure
                and offset.
            compute_hat_matrix: Whether to retain the full local smoother matrix.
            compute_inference: Whether to compute local standard errors and Wald
                statistics. Non-Gaussian diagnostics still retain trace statistics.
            compute_local_r2: Gaussian-only option forwarded to standard GWR.

        Returns:
            The fitted estimator.
        """
        self._reset_fit_state()
        self.family = self._normalize_family(self.family)
        if self.family == "gaussian":
            if exposure is not None or offset is not None:
                raise ValueError("Gaussian GWGLM does not use exposure or offset.")
            super().fit(
                X,
                y,
                coords,
                compute_hat_matrix=compute_hat_matrix,
                compute_inference=compute_inference,
                compute_local_r2=compute_local_r2,
            )
            self.family_ = "gaussian"
            self.mu_ = self.fitted_values_.copy()
            self.linear_predictor_ = self.fitted_values_.copy()
            self.exposure_train_ = np.ones(self.n_samples_, dtype=float)
            self.offset_train_ = np.zeros(self.n_samples_, dtype=float)
            self.deviance_residuals_ = self.residuals_.copy()
            sigma = np.sqrt(max(float(self.sigma2_ or 0.0), _EPS))
            self.pearson_residuals_ = self.residuals_ / sigma
            self.deviance_ = float(np.dot(self.residuals_, self.residuals_))
            null_residuals = self.y_train_ - np.mean(self.y_train_)
            self.null_deviance_ = float(np.dot(null_residuals, null_residuals))
            self.percent_deviance_ = (
                1.0 - self.deviance_ / self.null_deviance_
                if self.null_deviance_ > _EPS
                else np.nan
            )
            self.adjusted_percent_deviance_ = float(
                self.diagnostics_.get("adjusted_r2", np.nan)
            )
            self.log_likelihood_ = self._log_likelihood(self.y_train_, self.mu_)
            self.iteration_counts_ = np.ones(self.n_samples_, dtype=int)
            self.local_converged_ = np.ones(self.n_samples_, dtype=bool)
            self.converged_ = True
            self.final_working_weights_ = np.ones(self.n_samples_, dtype=float)
            self.parameter_z_values_ = (
                None
                if self.parameter_t_values_ is None
                else self.parameter_t_values_.copy()
            )
            self.intercept_z_ = (
                None if self.intercept_t_ is None else self.intercept_t_.copy()
            )
            self.coef_z_ = None if self.coef_t_ is None else self.coef_t_.copy()
            return self

        try:
            X_arr, y_arr, coords_arr = self._validate_inputs(X, y, coords)
            self._validate_response(y_arr)
            exposure_arr, offset_arr = self._prepare_exposure(
                y_arr.size, exposure=exposure, offset=offset
            )
            self._store_training_data(X_arr, y_arr, coords_arr, copy=True)
            self.exposure_train_ = exposure_arr.copy()
            self.offset_train_ = offset_arr.copy()
            X_design = add_intercept(X_arr) if self.fit_intercept else X_arr
            self.kernel_func_ = get_kernel_function(self.kernel)
            self.bandwidth_ = self._resolve_non_gaussian_bandwidth(
                X_design, y_arr, coords_arr, exposure_arr
            )
            if self.verbose:
                kind = "adaptive neighbours" if self.adaptive else "fixed distance"
                print(
                    f"Fitting {self.family.title()} GWGLM with {kind} "
                    f"bandwidth={self.bandwidth_}..."
                )
            fit_result = self._fit_non_gaussian_locations(
                X_design,
                y_arr,
                coords_arr,
                exposure_arr,
                self.bandwidth_,
                store_hat_matrix=bool(compute_hat_matrix),
            )
            if self.fit_intercept:
                self.intercept_ = fit_result.params[:, 0].copy()
                self.coef_ = fit_result.params[:, 1:].copy()
            else:
                self.intercept_ = np.zeros(y_arr.size, dtype=float)
                self.coef_ = fit_result.params.copy()
            self.family_ = self.family
            self.fitted_values_ = fit_result.fitted_values.copy()
            self.mu_ = self.fitted_values_.copy()
            self.linear_predictor_ = fit_result.linear_predictor.copy()
            self.residuals_ = y_arr - self.fitted_values_
            self.hat_matrix_ = fit_result.hat_matrix
            self.S_matrix_ = self.hat_matrix_
            self.iteration_counts_ = fit_result.iteration_counts.copy()
            self.local_converged_ = fit_result.converged.copy()
            self.converged_ = bool(np.all(self.local_converged_))
            self.final_working_weights_ = fit_result.final_working_weights.copy()
            self.deviance_residuals_ = self._deviance_residuals(y_arr, self.mu_)
            if self.family == "poisson":
                variance = np.clip(self.mu_, _EPS, None)
            else:
                variance = np.clip(self.mu_ * (1.0 - self.mu_), _EPS, None)
            self.pearson_residuals_ = self.residuals_ / np.sqrt(variance)
            self.diagnostics_ = self._information_diagnostics(
                y_arr,
                self.mu_,
                exposure_arr,
                trace_s=fit_result.trace_S,
                trace_sts=fit_result.trace_StS,
            )
            self.deviance_ = self.diagnostics_["deviance"]
            self.null_deviance_ = self.diagnostics_["null_deviance"]
            self.log_likelihood_ = self.diagnostics_["log_likelihood"]
            self.percent_deviance_ = self.diagnostics_["percent_deviance"]
            self.adjusted_percent_deviance_ = self.diagnostics_[
                "adjusted_percent_deviance"
            ]
            if compute_inference:
                self._set_non_gaussian_inference(fit_result, self.diagnostics_)
            else:
                self.influence_ = fit_result.influence.copy()
                self.inference_enabled_ = False
            self.local_r2_ = None
            if isinstance(self.bandwidth, str) or self.bandwidth is None:
                method = (
                    self.bandwidth.strip().lower()
                    if isinstance(self.bandwidth, str)
                    else self.bandwidth_method.strip().lower()
                )
                if method == "cv":
                    cv_score, cv_residuals = self._leave_one_out_score(
                        X_design,
                        y_arr,
                        coords_arr,
                        exposure_arr,
                        self.bandwidth_,
                        return_residuals=True,
                    )
                    self.bandwidth_selection_score_ = float(cv_score)
                    self.cv_residuals_ = cv_residuals.copy()
                    self.cv_contributions_ = cv_residuals**2
            self._mark_fitted()
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def _prediction_non_gaussian_parameters(
        self,
        coords: ArrayLike,
    ) -> Dict[str, np.ndarray]:
        self._check_is_fitted()
        if (
            self.X_train_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self.exposure_train_ is None
            or self.bandwidth_ is None
            or self.kernel_func_ is None
        ):
            raise RuntimeError("Stored GWGLM training state is incomplete.")
        coords_arr = validate_coords(coords)
        X_design = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        distances = compute_distance_matrix(
            coords_arr, self.coords_train_, metric=self.distance_metric
        )
        full_params = np.empty((coords_arr.shape[0], X_design.shape[1]), dtype=float)
        covariance_factors = np.empty_like(full_params)
        for index, distance_row in enumerate(distances):
            spatial_weights = self._candidate_weights(distance_row, self.bandwidth_)
            local = self._iwls(
                X_design,
                self.y_train_,
                spatial_weights,
                self.exposure_train_,
            )
            full_params[index] = local.params
            covariance_factors[index] = np.sum(
                local.inverse_xtx_xt * local.inverse_xtx_xt, axis=1
            )
        if self.fit_intercept:
            intercept = full_params[:, 0]
            coef = full_params[:, 1:]
        else:
            intercept = np.zeros(coords_arr.shape[0], dtype=float)
            coef = full_params
        standard_errors = np.sqrt(np.maximum(covariance_factors, 0.0))
        z_values = np.full_like(full_params, np.nan, dtype=float)
        np.divide(
            full_params,
            standard_errors,
            out=z_values,
            where=standard_errors > _EPS,
        )
        return {
            "coords": coords_arr,
            "coef": coef,
            "intercept": intercept,
            "standard_errors": standard_errors,
            "z_values": z_values,
        }

    def predict(
        self,
        X: ArrayLike,
        coords: ArrayLike,
        *,
        exposure: Optional[object] = None,
        offset: Optional[object] = None,
    ) -> np.ndarray:
        """Predict conditional means at target locations."""
        return self.predict_result(
            X, coords, exposure=exposure, offset=offset
        ).predictions

    def predict_result(
        self,
        X: ArrayLike,
        coords: ArrayLike,
        *,
        exposure: Optional[object] = None,
        offset: Optional[object] = None,
    ) -> Union[GWGLMPredictionResult, GWRPredictionResult]:
        """Return predictions, local parameters, and optional inference results."""
        if self.family_ == "gaussian":
            if exposure is not None or offset is not None:
                raise ValueError("Gaussian GWGLM does not use exposure or offset.")
            return super().predict_result(X, coords)
        X_arr, coords_arr = self._validate_prediction_inputs(X, coords)
        exposure_arr, offset_arr = self._prepare_exposure(
            X_arr.shape[0], exposure=exposure, offset=offset
        )
        params = self._prediction_non_gaussian_parameters(coords_arr)
        coef = params["coef"]
        intercept = params["intercept"]
        linear_predictor = np.einsum("ij,ij->i", X_arr, coef) + intercept
        if self.family_ == "poisson":
            predictions = np.exp(np.clip(linear_predictor + offset_arr, -700.0, 700.0))
            result_exposure: Optional[np.ndarray] = exposure_arr
        else:
            predictions = expit(linear_predictor)
            result_exposure = None
        standard_errors = params["standard_errors"]
        z_values = params["z_values"]
        if self.fit_intercept:
            intercept_se = standard_errors[:, 0]
            coef_se = standard_errors[:, 1:]
            intercept_z = z_values[:, 0]
            coef_z = z_values[:, 1:]
        else:
            intercept_se = np.zeros(X_arr.shape[0], dtype=float)
            coef_se = standard_errors
            intercept_z = np.full(X_arr.shape[0], np.nan, dtype=float)
            coef_z = z_values
        names = (
            tuple(str(name) for name in self.feature_names_in_)
            if self.feature_names_in_ is not None
            else tuple(f"x{index}" for index in range(X_arr.shape[1]))
        )
        return GWGLMPredictionResult(
            predictions=np.asarray(predictions, dtype=float),
            linear_predictor=np.asarray(linear_predictor, dtype=float),
            coef=np.asarray(coef, dtype=float),
            intercept=np.asarray(intercept, dtype=float),
            coords=np.asarray(coords_arr, dtype=float),
            feature_names=names,
            family=str(self.family_),
            exposure=result_exposure,
            coef_standard_errors=coef_se,
            intercept_standard_errors=intercept_se,
            coef_z_values=coef_z,
            intercept_z_values=intercept_z,
        )

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        coords: ArrayLike,
        *,
        exposure: Optional[object] = None,
        offset: Optional[object] = None,
    ) -> float:
        """Return R² for Gaussian models or deviance explained otherwise."""
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        predictions = self.predict(X, coords, exposure=exposure, offset=offset)
        if self.family_ == "gaussian":
            residual = y_arr - predictions
            total = y_arr - np.mean(y_arr)
            denominator = float(np.dot(total, total))
            return 1.0 - float(np.dot(residual, residual)) / denominator
        exposure_arr, _ = self._prepare_exposure(
            y_arr.size, exposure=exposure, offset=offset
        )
        deviance = self._deviance(y_arr, predictions)
        null_deviance = self._deviance(y_arr, self._null_mean(y_arr, exposure_arr))
        return 1.0 - deviance / null_deviance

    def to_frame(self) -> pd.DataFrame:
        """Return training-location parameters and GLM diagnostics."""
        frame = super().to_frame()
        for name, values in (
            ("linear_predictor", self.linear_predictor_),
            ("deviance_residual", self.deviance_residuals_),
            ("pearson_residual", self.pearson_residuals_),
            ("influence", self.influence_),
            ("iteration_count", self.iteration_counts_),
            ("local_converged", self.local_converged_),
        ):
            if values is not None:
                frame[name] = np.asarray(values).reshape(-1)
        if self.family_ == "poisson" and self.exposure_train_ is not None:
            frame["exposure"] = self.exposure_train_
            frame["offset"] = self.offset_train_
        feature_names = (
            [str(name) for name in self.feature_names_in_]
            if self.feature_names_in_ is not None
            else [f"x{index}" for index in range(self.n_features_in_ or 0)]
        )
        if self.intercept_se_ is not None:
            frame["intercept_se"] = self.intercept_se_
        if self.intercept_z_ is not None:
            frame["intercept_z"] = self.intercept_z_
        if self.coef_se_ is not None:
            for index, name in enumerate(feature_names):
                frame[f"se_{name}"] = self.coef_se_[:, index]
        if self.coef_z_ is not None:
            for index, name in enumerate(feature_names):
                frame[f"z_{name}"] = self.coef_z_[:, index]
        return frame

    def summary(self) -> str:
        """Return a stable text summary of GWGLM results."""
        self._check_is_fitted()
        if self.family_ == "gaussian":
            return (
                super()
                .summary()
                .replace(
                    "Gaussian Geographically Weighted Regression (GWR)",
                    "Gaussian Geographically Weighted Generalized Linear Model (GWGLM)",
                    1,
                )
            )
        if self.diagnostics_ is None:
            raise RuntimeError("GWGLM diagnostics are unavailable.")
        lines = [
            "=" * 78,
            f"{self.family_.title()} Geographically Weighted GLM (GWGLM)",
            "=" * 78,
            f"Samples: {self.n_samples_}",
            f"Predictors: {self.n_features_in_}",
            f"Kernel: {self.kernel}",
            f"Bandwidth: {self.bandwidth_} ({'adaptive neighbours' if self.adaptive else 'fixed distance'})",
            f"Distance metric: {self.distance_metric}",
            f"All local fits converged: {self.converged_}",
            f"Maximum local IWLS iterations: {int(np.max(self.iteration_counts_))}",
            "",
            "Model diagnostics",
            "-" * 78,
            f"Deviance: {self.deviance_:.6f}",
            f"Null deviance: {self.null_deviance_:.6f}",
            f"Deviance explained: {self.percent_deviance_:.6f}",
            f"Adjusted deviance explained: {self.adjusted_percent_deviance_:.6f}",
            f"Effective parameters (trace(S)): {self.diagnostics_['effective_params']:.6f}",
            f"AIC: {self.diagnostics_['aic']:.6f}",
            f"AICc: {self.diagnostics_['aicc']:.6f}",
            f"BIC: {self.diagnostics_['bic']:.6f}",
            "=" * 78,
        ]
        return "\n".join(lines)
