# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Geographically weighted summary statistics.

This module implements the moment- and order-based local descriptive statistics
provided by ``GWmodel::gwss`` using the same kernel, covariance, correlation, and
weighted-quantile definitions.

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
from scipy.stats import rankdata

from pygwrx.core._summary import format_summary
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import compute_distance_matrix, validate_coords


class GWSS:
    """Compute geographically weighted summary statistics.

    Args:
        kernel: Kernel name or callable accepted by
            :func:`pygwrx.core.kernels.get_kernel_function`.
        bandwidth: Positive fixed distance or, when ``adaptive=True``, a positive
            integer number of nearest neighbours. If ``None``, a leave-one-out
            cross-validation bandwidth for local means is selected during fitting.
        adaptive: Whether ``bandwidth`` represents a nearest-neighbour count.
        quantile: Whether to calculate local median, interquartile range, and
            quantile imbalance.
        verbose: Whether to print a compact completion message.

    Notes:
        Moment statistics follow ``GWmodel::gwss``. In particular, local variance
        is the normalized weighted second central moment, whereas bivariate
        covariance uses the unbiased ``stats::cov.wt`` denominator
        ``1 - sum(w**2)``. Weighted quantiles reproduce GWmodel's ``findq`` rule.
    """

    def __init__(
        self,
        kernel: str | Any = "bisquare",
        bandwidth: float | int | None = None,
        adaptive: bool = False,
        quantile: bool = False,
        verbose: bool = False,
    ) -> None:
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.adaptive = bool(adaptive)
        self.quantile = bool(quantile)
        self.verbose = bool(verbose)
        get_kernel_function(kernel)
        if bandwidth is not None:
            self._validate_bandwidth(bandwidth, n_samples=None)
        self._clear_fit_state()

    def _clear_fit_state(self) -> None:
        """Clear all fitted attributes before a new calibration attempt."""
        names = (
            "local_mean_",
            "local_std_",
            "local_var_",
            "local_skewness_",
            "local_cv_",
            "local_median_",
            "local_iqr_",
            "local_qi_",
            "local_cov_",
            "local_corr_",
            "local_corr_spearman_",
            "X_data_",
            "coords_data_",
            "coords_summary_",
            "bandwidth_",
            "var_names_",
            "weights_",
        )
        for name in names:
            setattr(self, name, None)
        self._is_fitted = False

    def _validate_bandwidth(
        self, bandwidth: float | int, n_samples: int | None
    ) -> float | int:
        """Validate fixed or adaptive bandwidth semantics."""
        if isinstance(bandwidth, (bool, np.bool_)):
            raise TypeError("bandwidth must not be bool.")
        if self.adaptive:
            if not isinstance(bandwidth, Integral):
                raise TypeError(
                    "adaptive bandwidth must be an integer neighbour count."
                )
            value = int(bandwidth)
            if value < 2:
                raise ValueError("adaptive bandwidth must be at least 2.")
            if n_samples is not None and value > n_samples:
                raise ValueError("adaptive bandwidth cannot exceed the sample size.")
            return value
        if not isinstance(bandwidth, Real):
            raise TypeError("fixed bandwidth must be a positive real scalar.")
        value = float(bandwidth)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("fixed bandwidth must be finite and greater than zero.")
        return value

    @staticmethod
    def _as_data_matrix(X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Convert and validate a numeric two-dimensional data matrix."""
        if isinstance(X, pd.DataFrame):
            names = [str(column) for column in X.columns]
            raw = X.to_numpy()
        else:
            raw = X
            array = np.asarray(raw)
            if array.ndim == 1:
                names = ["Var_0"]
            elif array.ndim == 2:
                names = [f"Var_{i}" for i in range(array.shape[1])]
            else:
                names = []
        try:
            array = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must contain only numeric values.") from exc
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
            raise ValueError(
                "X must be a two-dimensional matrix with at least two rows."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("X contains NaN or infinite values.")
        return array, names

    def _weights(self, distances: np.ndarray, bandwidth: float | int) -> np.ndarray:
        """Construct fixed or GWmodel-compatible adaptive kernel weights."""
        kernel = get_kernel_function(self.kernel)
        if not self.adaptive:
            return np.asarray(kernel(distances, float(bandwidth)), dtype=float)
        k = int(bandwidth)
        order = np.argsort(distances, kind="stable")
        kernel_name = (
            self.kernel.strip().lower() if isinstance(self.kernel, str) else None
        )
        if kernel_name == "boxcar":
            weights = np.zeros_like(distances, dtype=float)
            weights[order[:k]] = 1.0
            return weights
        kth_distance = float(distances[order[k - 1]])
        if kth_distance == 0.0:
            weights = np.zeros_like(distances, dtype=float)
            weights[order[:k]] = 1.0
            return weights
        return np.asarray(kernel(distances, kth_distance), dtype=float)

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray, weights: np.ndarray, probs: tuple[float, ...]
    ) -> np.ndarray:
        """Reproduce the weighted-quantile rule used by ``GWmodel::gwss``."""
        order = np.argsort(values, kind="stable")
        x = values[order]
        cumulative = np.cumsum(weights[order])
        result = []
        for probability in probs:
            eligible = np.flatnonzero(cumulative <= probability)
            index = int(eligible[-1]) if eligible.size else 0
            result.append(x[index])
        return np.asarray(result, dtype=float)

    @staticmethod
    def _unbiased_weighted_covariance(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> float:
        """Match ``stats::cov.wt(..., method='unbiased')`` for normalized weights."""
        mx = float(w @ x)
        my = float(w @ y)
        denominator = 1.0 - float(w @ w)
        if denominator <= np.finfo(float).eps:
            return np.nan
        return float(w @ ((x - mx) * (y - my)) / denominator)

    def select_bandwidth(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
        *,
        statistic: str = "mean",
    ) -> float | int:
        """Select a shared bandwidth by leave-one-out CV.

        The score sums the GWmodel mean- or median-CV scores over all variables.
        This method returns one shared bandwidth suitable for ``gwss``; GWmodel's
        ``bw.gwss.average`` additionally reports variable-specific bandwidths.
        """
        X_arr, _ = self._as_data_matrix(X)
        coords_arr = validate_coords(coords)
        if coords_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        statistic_name = str(statistic).strip().lower()
        if statistic_name not in {"mean", "median"}:
            raise ValueError("statistic must be 'mean' or 'median'.")
        distances = compute_distance_matrix(coords_arr, coords_arr)

        def score(candidate: float | int) -> float:
            bw = int(round(candidate)) if self.adaptive else float(candidate)
            total = 0.0
            for i in range(X_arr.shape[0]):
                raw = self._weights(distances[i], bw)
                if not np.isfinite(raw).all() or raw.sum() <= 0:
                    return np.inf
                full = raw / raw.sum()
                leave = raw.copy()
                leave[i] = 0.0
                if leave.sum() <= 0:
                    return np.inf
                leave /= leave.sum()
                if statistic_name == "mean":
                    difference = full @ X_arr - leave @ X_arr
                else:
                    full_m = np.array(
                        [
                            self._weighted_quantile(X_arr[:, j], full, (0.5,))[0]
                            for j in range(X_arr.shape[1])
                        ]
                    )
                    keep = np.arange(X_arr.shape[0]) != i
                    leave_m = np.array(
                        [
                            self._weighted_quantile(
                                X_arr[keep, j], leave[keep], (0.5,)
                            )[0]
                            for j in range(X_arr.shape[1])
                        ]
                    )
                    difference = full_m - leave_m
                total += float(difference @ difference)
            return total

        n = X_arr.shape[0]
        if self.adaptive:
            lower = 2
            candidates = range(lower, n + 1)
            scores = [(candidate, score(candidate)) for candidate in candidates]
            return min(scores, key=lambda item: (item[1], item[0]))[0]
        upper = float(np.max(distances))
        if upper <= 0:
            raise ValueError(
                "fixed-bandwidth selection requires non-identical coordinates."
            )
        result = minimize_scalar(
            score, bounds=(upper / 5000.0, upper), method="bounded"
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("bandwidth selection failed.")
        return float(result.x)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
        summary_coords: np.ndarray | pd.DataFrame | None = None,
    ) -> "GWSS":
        """Calculate local summary statistics and store the fitted result."""
        self._clear_fit_state()
        try:
            X_arr, names = self._as_data_matrix(X)
            coords_arr = validate_coords(coords)
            if coords_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("X and coords must contain the same number of rows.")
            summary_arr = (
                coords_arr
                if summary_coords is None
                else validate_coords(summary_coords)
            )
            if summary_arr.shape[1] != coords_arr.shape[1]:
                raise ValueError(
                    "summary_coords and coords must have the same dimension."
                )
            bandwidth = (
                self.select_bandwidth(X_arr, coords_arr)
                if self.bandwidth is None
                else self._validate_bandwidth(self.bandwidth, X_arr.shape[0])
            )
            distances = compute_distance_matrix(summary_arr, coords_arr)
            weight_rows = []
            for i in range(summary_arr.shape[0]):
                raw = self._weights(distances[i], bandwidth)
                if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
                    raise ValueError(
                        f"kernel weights are undefined at summary location {i}."
                    )
                weight_rows.append(raw / raw.sum())
            W = np.vstack(weight_rows)

            means = W @ X_arr
            centered = X_arr[None, :, :] - means[:, None, :]
            variances = np.einsum("sn,snv->sv", W, centered**2)
            std = np.sqrt(np.maximum(variances, 0.0))
            third = np.einsum("sn,snv->sv", W, centered**3)
            skew = np.divide(
                third, std**3, out=np.full_like(third, np.nan), where=std > 0
            )
            cv = np.divide(std, means, out=np.full_like(std, np.nan), where=means != 0)

            medians = iqrs = qis = None
            if self.quantile:
                quantiles = np.empty((summary_arr.shape[0], X_arr.shape[1], 3))
                for i, w in enumerate(W):
                    for j in range(X_arr.shape[1]):
                        quantiles[i, j] = self._weighted_quantile(
                            X_arr[:, j], w, (0.25, 0.5, 0.75)
                        )
                medians = quantiles[:, :, 1]
                iqrs = quantiles[:, :, 2] - quantiles[:, :, 0]
                numerator = 2 * medians - quantiles[:, :, 2] - quantiles[:, :, 0]
                qis = np.divide(
                    numerator, iqrs, out=np.full_like(iqrs, np.nan), where=iqrs != 0
                )

            covariances: dict[tuple[int, int], np.ndarray] = {}
            correlations: dict[tuple[int, int], np.ndarray] = {}
            rank_correlations: dict[tuple[int, int], np.ndarray] = {}
            ranks = np.column_stack(
                [rankdata(X_arr[:, j], method="average") for j in range(X_arr.shape[1])]
            )
            for j in range(X_arr.shape[1] - 1):
                for k in range(j + 1, X_arr.shape[1]):
                    cov = np.array(
                        [
                            self._unbiased_weighted_covariance(
                                X_arr[:, j], X_arr[:, k], w
                            )
                            for w in W
                        ]
                    )
                    var_j = np.array(
                        [
                            self._unbiased_weighted_covariance(
                                X_arr[:, j], X_arr[:, j], w
                            )
                            for w in W
                        ]
                    )
                    var_k = np.array(
                        [
                            self._unbiased_weighted_covariance(
                                X_arr[:, k], X_arr[:, k], w
                            )
                            for w in W
                        ]
                    )
                    denom = np.sqrt(var_j * var_k)
                    corr = np.divide(
                        cov, denom, out=np.full_like(cov, np.nan), where=denom > 0
                    )
                    rcov = np.array(
                        [
                            self._unbiased_weighted_covariance(
                                ranks[:, j], ranks[:, k], w
                            )
                            for w in W
                        ]
                    )
                    rvar_j = np.array(
                        [
                            self._unbiased_weighted_covariance(
                                ranks[:, j], ranks[:, j], w
                            )
                            for w in W
                        ]
                    )
                    rvar_k = np.array(
                        [
                            self._unbiased_weighted_covariance(
                                ranks[:, k], ranks[:, k], w
                            )
                            for w in W
                        ]
                    )
                    rdenom = np.sqrt(rvar_j * rvar_k)
                    rho = np.divide(
                        rcov, rdenom, out=np.full_like(rcov, np.nan), where=rdenom > 0
                    )
                    covariances[(j, k)] = cov
                    correlations[(j, k)] = corr
                    rank_correlations[(j, k)] = rho

            self.X_data_ = X_arr
            self.coords_data_ = coords_arr
            self.coords_summary_ = summary_arr
            self.bandwidth_ = bandwidth
            self.var_names_ = names
            self.weights_ = W
            self.local_mean_ = means
            self.local_var_ = variances
            self.local_std_ = std
            self.local_skewness_ = skew
            self.local_cv_ = cv
            self.local_median_ = medians
            self.local_iqr_ = iqrs
            self.local_qi_ = qis
            self.local_cov_ = covariances
            self.local_corr_ = correlations
            self.local_corr_spearman_ = rank_correlations
            self._is_fitted = True
        except Exception:
            self._clear_fit_state()
            raise
        if self.verbose:
            print(f"GWSS fitted at {self.coords_summary_.shape[0]} locations.")
        return self

    def summary(self) -> str:
        """Return a plain-text summary of the fitted local statistics."""
        self._require_fitted()
        result: dict[str, Any] = {
            "n_vars": len(self.var_names_),
            "var_names": list(self.var_names_),
            "n_summary_locations": self.local_mean_.shape[0],
            "bandwidth": self.bandwidth_,
            "adaptive": self.adaptive,
        }
        for i, name in enumerate(self.var_names_):
            result[f"{name}_mean_range"] = (
                float(np.nanmin(self.local_mean_[:, i])),
                float(np.nanmax(self.local_mean_[:, i])),
            )
            result[f"{name}_std_range"] = (
                float(np.nanmin(self.local_std_[:, i])),
                float(np.nanmax(self.local_std_[:, i])),
            )
        return format_summary("GWSS Summary", result)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return local statistics in GWmodel-compatible column naming."""
        self._require_fitted()
        data: dict[str, np.ndarray] = {
            "x": self.coords_summary_[:, 0],
            "y": self.coords_summary_[:, 1],
        }
        for i, name in enumerate(self.var_names_):
            data[f"{name}_LM"] = self.local_mean_[:, i]
            data[f"{name}_LSD"] = self.local_std_[:, i]
            data[f"{name}_LVar"] = self.local_var_[:, i]
            data[f"{name}_LSKe"] = self.local_skewness_[:, i]
            data[f"{name}_LCV"] = self.local_cv_[:, i]
            if self.quantile:
                data[f"{name}_Median"] = self.local_median_[:, i]
                data[f"{name}_IQR"] = self.local_iqr_[:, i]
                data[f"{name}_QI"] = self.local_qi_[:, i]
        for (i, j), cov in self.local_cov_.items():
            left, right = self.var_names_[i], self.var_names_[j]
            data[f"Cov_{left}.{right}"] = cov
            data[f"Corr_{left}.{right}"] = self.local_corr_[(i, j)]
            data[f"Spearman_rho_{left}.{right}"] = self.local_corr_spearman_[(i, j)]
        return pd.DataFrame(data)
