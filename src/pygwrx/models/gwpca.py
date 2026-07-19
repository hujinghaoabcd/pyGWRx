# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Geographically weighted principal component analysis.

This module implements basic GWPCA using the globally preprocessed and locally
weighted singular-value decomposition defined by Harris et al. (2011) and the
maintained ``GWmodel::gwpca`` implementation.

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

from pygwrx._optional import import_optional_dependency
from pygwrx.core._summary import format_summary
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import compute_distance_matrix, validate_coords


class GWPCA:
    r"""Fit a basic geographically weighted principal component analysis.

    Args:
        n_components: Number of local principal components to retain.
        kernel: Spatial kernel name or callable accepted by
            :func:`pygwrx.core.kernels.get_kernel_function`.
        bandwidth: Positive fixed distance or, when ``adaptive=True``, a
            positive integer neighbour count. ``None`` or ``"cv"`` selects a
            bandwidth using GWmodel-compatible leave-one-out cross-validation
            and its golden-section search.
        adaptive: Whether the bandwidth represents a nearest-neighbour count.
        scaling: Whether to globally standardize variables before local PCA.
            When false, variables are globally centered only. Both paths then
            apply local weighted centering, matching GWmodel.
        compute_scores: Whether to retain locally centered scores for all
            observations receiving positive weight at each evaluation location.
        verbose: Whether to print a compact completion message.

    Notes:
        At evaluation location :math:`u`, basic GWPCA decomposes

        .. math::

            \sqrt{W(u)}\{X^* - \bar{X}_w(u)\} = U D V^T,

        where :math:`X^*` is the globally centered or standardized matrix.
        Local loadings are columns of :math:`V`, and local component variances
        are :math:`D^2 / \sum_i w_i(u)`.

        Principal-component signs are mathematically indeterminate. pyGWRx
        applies a deterministic convention: the largest absolute loading in
        each component is made positive.
    """

    def __init__(
        self,
        n_components: int = 2,
        kernel: str | Any = "bisquare",
        bandwidth: float | int | str | None = "cv",
        adaptive: bool = True,
        scaling: bool = True,
        compute_scores: bool = False,
        verbose: bool = False,
    ) -> None:
        if isinstance(n_components, (bool, np.bool_)) or not isinstance(
            n_components, Integral
        ):
            raise TypeError("n_components must be a positive integer.")
        if int(n_components) < 1:
            raise ValueError("n_components must be at least 1.")
        if not isinstance(adaptive, (bool, np.bool_)):
            raise TypeError("adaptive must be boolean.")
        if not isinstance(scaling, (bool, np.bool_)):
            raise TypeError("scaling must be boolean.")
        if not isinstance(compute_scores, (bool, np.bool_)):
            raise TypeError("compute_scores must be boolean.")
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be boolean.")
        if isinstance(bandwidth, str):
            name = bandwidth.strip().lower()
            if name not in {"cv"}:
                if name == "adaptive":
                    raise ValueError(
                        "Use adaptive=True with bandwidth='cv' or an integer "
                        "neighbour count; bandwidth='adaptive' is not supported."
                    )
                raise ValueError("bandwidth string must be 'cv'.")
        get_kernel_function(kernel)

        self.n_components = int(n_components)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.adaptive = bool(adaptive)
        self.scaling = bool(scaling)
        self.compute_scores = bool(compute_scores)
        self.verbose = bool(verbose)
        if bandwidth is not None and not isinstance(bandwidth, str):
            self._validate_bandwidth(bandwidth, n_samples=None)
        self._clear_fit_state()

    def _clear_fit_state(self) -> None:
        """Clear fitted state before every calibration attempt."""
        names = (
            "loadings_",
            "var_",
            "local_pv_",
            "cumulative_pv_",
            "scores_",
            "focal_scores_",
            "cv_scores_",
            "pca_global_",
            "X_train_",
            "X_processed_",
            "coords_train_",
            "eval_coords_",
            "bandwidth_",
            "local_means_",
            "global_mean_",
            "global_scale_",
            "feature_names_",
            "weights_",
        )
        for name in names:
            setattr(self, name, None)
        self._is_fitted = False

    @staticmethod
    def _as_data_matrix(
        X: np.ndarray | pd.DataFrame,
        *,
        min_rows: int = 2,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert input to a finite two-dimensional numeric matrix."""
        if isinstance(X, pd.DataFrame):
            names = [str(column) for column in X.columns]
            raw = X.to_numpy()
        else:
            raw = X
            array = np.asarray(raw)
            names = (
                [f"Var_{index}" for index in range(array.shape[1])]
                if array.ndim == 2
                else []
            )
        try:
            array = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must contain only numeric values.") from exc
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")
        if array.shape[0] < min_rows:
            raise ValueError(
                f"X must contain at least {min_rows} observation"
                f"{'s' if min_rows != 1 else ''}."
            )
        if array.shape[1] < 2:
            raise ValueError("GWPCA requires at least two variables.")
        if not np.all(np.isfinite(array)):
            raise ValueError("X contains NaN or infinite values.")
        return array, names

    def _validate_bandwidth(
        self, bandwidth: float | int, n_samples: int | None
    ) -> float | int:
        """Validate fixed-distance or adaptive-neighbour bandwidth semantics."""
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

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        """Apply GWmodel-compatible global centering or standardization."""
        mean = np.mean(X, axis=0)
        if self.scaling:
            scale = np.std(X, axis=0, ddof=1)
            if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(
                    "scaling=True requires every variable to have positive "
                    "finite sample standard deviation."
                )
        else:
            scale = np.ones(X.shape[1], dtype=float)
        self.global_mean_ = mean
        self.global_scale_ = scale
        return (X - mean) / scale

    def _preprocess_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted global preprocessing to new rows."""
        if X.shape[1] != self.global_mean_.shape[0]:
            raise ValueError("X has a different number of variables than the fit data.")
        return (X - self.global_mean_) / self.global_scale_

    @staticmethod
    def _canonicalize_loadings(loadings: np.ndarray) -> np.ndarray:
        """Apply a deterministic sign convention to loading vectors."""
        result = np.asarray(loadings, dtype=float).copy()
        for component in range(result.shape[1]):
            column = result[:, component]
            pivot = int(np.argmax(np.abs(column)))
            if column[pivot] < 0:
                result[:, component] *= -1.0
        return result

    def _local_pca(
        self,
        X: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute one basic weighted PCA using the GWmodel SVD definition."""
        positive = np.asarray(weights, dtype=float) > 0
        X_local = X[positive]
        w_local = np.asarray(weights, dtype=float)[positive]
        minimum = max(self.n_components + 1, 2)
        if X_local.shape[0] < minimum:
            raise ValueError(
                "The bandwidth supplies too few positively weighted observations "
                f"for {self.n_components} retained components."
            )
        sum_w = float(np.sum(w_local))
        if not np.isfinite(sum_w) or sum_w <= 0:
            raise ValueError("Local kernel weights must have a positive finite sum.")

        local_mean = np.sum(X_local * w_local[:, None], axis=0) / sum_w
        centered = X_local - local_mean
        weighted = centered * np.sqrt(w_local)[:, None]
        try:
            _, singular_values, vt = np.linalg.svd(weighted, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError("Local weighted SVD failed.") from exc

        loadings = self._canonicalize_loadings(vt.T[:, : self.n_components])
        variances = np.zeros(X.shape[1], dtype=float)
        available = min(singular_values.size, variances.size)
        variances[:available] = singular_values[:available] ** 2 / sum_w
        local_scores = centered @ loadings
        return loadings, variances, local_mean, local_scores

    def _cv_contributions(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        bandwidth: float | int,
    ) -> np.ndarray:
        r"""Return GWmodel-compatible leave-one-out CV contributions.

        GWmodel computes each contribution as

        .. math::

            \left[\sum_j \{x_{ij} - \hat{x}_{ij}^{(-i)}\}\right]^2,

        rather than the more usual sum of squared component-wise
        reconstruction residuals. This definition is retained deliberately so
        that public GWmodel bandwidth benchmarks, including the Dublin voter
        example, are reproduced exactly.
        """
        distances = compute_distance_matrix(coords, coords)
        contributions = np.empty(X.shape[0], dtype=float)
        for index in range(X.shape[0]):
            weights = self._weights(distances[index], bandwidth)
            weights[index] = 0.0
            try:
                loadings, _, _, _ = self._local_pca(X, weights)
            except (ValueError, np.linalg.LinAlgError):
                contributions[index] = np.inf
                continue
            projection = loadings @ loadings.T
            residual = X[index] - X[index] @ projection
            contributions[index] = float(np.sum(residual) ** 2)
        return contributions

    @staticmethod
    def _gwmodel_golden_search(
        score_function: Any,
        lower: float,
        upper: float,
        *,
        adaptive: bool,
    ) -> float | int:
        """Replicate GWmodel's ``gold`` bandwidth-search routine.

        The adaptive path intentionally follows GWmodel's alternating
        ``floor``/``round`` integer updates. This matters for published
        benchmarks and is not equivalent to exhaustive discrete minimization.
        """
        epsilon = 1.0e-4
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        distance = ratio * (upper - lower)
        if adaptive:
            point_1 = int(np.floor(lower + distance))
            point_2 = int(np.rint(upper - distance))
        else:
            point_1 = lower + distance
            point_2 = upper - distance

        value_1 = float(score_function(point_1))
        value_2 = float(score_function(point_2))
        difference = value_2 - value_1
        optimum: float | int = point_1 if value_1 < value_2 else point_2

        while abs(distance) > epsilon and abs(difference) > epsilon:
            distance *= ratio
            if value_1 < value_2:
                lower = point_2
                point_2 = point_1
                point_1 = (
                    int(np.rint(lower + distance)) if adaptive else lower + distance
                )
                value_2 = value_1
                value_1 = float(score_function(point_1))
            else:
                upper = point_1
                point_1 = point_2
                point_2 = (
                    int(np.floor(upper - distance)) if adaptive else upper - distance
                )
                value_1 = value_2
                value_2 = float(score_function(point_2))
            optimum = point_1 if value_1 < value_2 else point_2
            difference = value_2 - value_1

        return int(optimum) if adaptive else float(optimum)

    def select_bandwidth(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
    ) -> float | int:
        """Select a fixed or adaptive bandwidth by leave-one-out CV."""
        X_arr, _ = self._as_data_matrix(X)
        coords_arr = validate_coords(coords)
        if coords_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        if self.n_components > X_arr.shape[1]:
            raise ValueError("n_components cannot exceed the number of variables.")

        mean = np.mean(X_arr, axis=0)
        if self.scaling:
            scale = np.std(X_arr, axis=0, ddof=1)
            if np.any(scale <= 0) or np.any(~np.isfinite(scale)):
                raise ValueError(
                    "scaling=True requires every variable to have positive "
                    "finite sample standard deviation."
                )
        else:
            scale = np.ones(X_arr.shape[1], dtype=float)
        processed = (X_arr - mean) / scale
        distances = compute_distance_matrix(coords_arr, coords_arr)

        score_cache: dict[float | int, float] = {}

        def score(candidate: float | int) -> float:
            bandwidth: float | int
            if self.adaptive:
                bandwidth = int(candidate)
            else:
                bandwidth = float(candidate)
            if bandwidth not in score_cache:
                values = self._cv_contributions(processed, coords_arr, bandwidth)
                score_cache[bandwidth] = (
                    float(np.sum(values)) if np.all(np.isfinite(values)) else np.inf
                )
            return score_cache[bandwidth]

        n_samples = X_arr.shape[0]
        if self.adaptive:
            selected = self._gwmodel_golden_search(
                score, 2.0, float(n_samples), adaptive=True
            )
            if not np.isfinite(score(selected)):
                raise RuntimeError("adaptive GWPCA bandwidth selection failed.")
            return int(selected)

        upper = float(np.max(distances))
        if upper <= 0:
            raise ValueError(
                "fixed-bandwidth selection requires non-identical coordinates."
            )
        selected = self._gwmodel_golden_search(
            score, upper / 5000.0, upper, adaptive=False
        )
        if not np.isfinite(score(selected)):
            raise RuntimeError("fixed GWPCA bandwidth selection failed.")
        return float(selected)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame,
        eval_coords: np.ndarray | pd.DataFrame | None = None,
        compute_cv: bool = False,
    ) -> "GWPCA":
        """Fit local principal components at observation or evaluation locations.

        Args:
            X: Numeric matrix with observations in rows and variables in columns.
            coords: Observation coordinates with the same row count as ``X``.
            eval_coords: Optional coordinates at which local loadings are
                evaluated. The observations in ``X`` remain the weighted data.
            compute_cv: Whether to retain leave-one-out reconstruction-error
                contributions for the selected or supplied bandwidth.

        Returns:
            GWPCA: The fitted estimator.

        Raises:
            ValueError: If inputs, bandwidth, or local windows are invalid.
        """
        self._clear_fit_state()
        X_arr, names = self._as_data_matrix(X)
        coords_arr = validate_coords(coords)
        if coords_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        if self.n_components > X_arr.shape[1]:
            raise ValueError("n_components cannot exceed the number of variables.")
        if not isinstance(compute_cv, (bool, np.bool_)):
            raise TypeError("compute_cv must be boolean.")

        if eval_coords is None:
            eval_arr = coords_arr.copy()
        else:
            eval_arr = validate_coords(eval_coords)
            if eval_arr.shape[1] != coords_arr.shape[1]:
                raise ValueError(
                    "eval_coords and coords must have the same coordinate dimension."
                )

        decomposition = import_optional_dependency(
            "sklearn.decomposition", extra="ml", purpose="GWPCA"
        )
        processed = self._preprocess_fit(X_arr)
        self.pca_global_ = decomposition.PCA(n_components=min(X_arr.shape)).fit(
            processed
        )

        if self.bandwidth is None or isinstance(self.bandwidth, str):
            bandwidth = self.select_bandwidth(X_arr, coords_arr)
        else:
            bandwidth = self._validate_bandwidth(
                self.bandwidth, n_samples=X_arr.shape[0]
            )

        distances = compute_distance_matrix(eval_arr, coords_arr)
        n_eval = eval_arr.shape[0]
        n_features = X_arr.shape[1]
        loadings = np.empty((n_eval, n_features, self.n_components), dtype=float)
        variances = np.empty((n_eval, n_features), dtype=float)
        local_means = np.empty((n_eval, n_features), dtype=float)
        weights_all = np.empty((n_eval, X_arr.shape[0]), dtype=float)
        local_scores: list[np.ndarray] | None = [] if self.compute_scores else None

        for index in range(n_eval):
            weights = self._weights(distances[index], bandwidth)
            local_loading, local_var, local_mean, score_matrix = self._local_pca(
                processed, weights
            )
            loadings[index] = local_loading
            variances[index] = local_var
            local_means[index] = local_mean
            weights_all[index] = weights
            if local_scores is not None:
                local_scores.append(score_matrix)

        total_variance = np.sum(variances, axis=1)
        if np.any(~np.isfinite(total_variance)) or np.any(total_variance <= 0):
            raise ValueError("At least one local PCA has zero or invalid variance.")
        local_pv = variances[:, : self.n_components] / total_variance[:, None] * 100.0

        self.X_train_ = X_arr.copy()
        self.X_processed_ = processed
        self.coords_train_ = coords_arr.copy()
        self.eval_coords_ = eval_arr.copy()
        self.bandwidth_ = bandwidth
        self.feature_names_ = names
        self.loadings_ = loadings
        self.var_ = variances
        self.local_means_ = local_means
        self.local_pv_ = local_pv
        self.cumulative_pv_ = np.sum(local_pv, axis=1)
        self.scores_ = local_scores
        self.weights_ = weights_all
        self.focal_scores_ = None
        if eval_coords is None:
            self.focal_scores_ = np.einsum(
                "ij,ijk->ik", processed - local_means, loadings
            )
        self.cv_scores_ = (
            self._cv_contributions(processed, coords_arr, bandwidth)
            if compute_cv
            else None
        )
        self._is_fitted = True

        if self.verbose:
            print(
                "GWPCA fit complete: "
                f"n={X_arr.shape[0]}, p={n_features}, "
                f"components={self.n_components}, bandwidth={bandwidth}."
            )
        return self

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
        coords: np.ndarray | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Project rows using loadings already calibrated at matching locations.

        This method does not interpolate or borrow the nearest loading surface.
        To score new locations, first fit with those locations in ``eval_coords``.

        Args:
            X: Rows to project, one row per fitted evaluation location.
            coords: Optional coordinates identifying the fitted evaluation
                locations. Every row must exactly match one fitted location.

        Returns:
            ndarray: Locally centered component scores.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        X_arr, _ = self._as_data_matrix(X, min_rows=1)
        processed = self._preprocess_transform(X_arr)

        if coords is None:
            if X_arr.shape[0] != self.eval_coords_.shape[0]:
                raise ValueError(
                    "Without coords, X must contain one row per fitted "
                    "evaluation location."
                )
            indices = np.arange(X_arr.shape[0])
        else:
            coords_arr = validate_coords(coords)
            if coords_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("X and coords must contain the same number of rows.")
            indices = []
            for coordinate in coords_arr:
                matches = np.flatnonzero(
                    np.all(np.isclose(self.eval_coords_, coordinate), axis=1)
                )
                if matches.size != 1:
                    raise ValueError(
                        "Every transform coordinate must match exactly one fitted "
                        "evaluation location; refit with the desired eval_coords."
                    )
                indices.append(int(matches[0]))
            indices = np.asarray(indices, dtype=int)

        centered = processed - self.local_means_[indices]
        return np.einsum("ij,ijk->ik", centered, self.loadings_[indices])

    def get_winning_variable(self, component: int = 0) -> np.ndarray:
        """Return the largest-absolute-loading variable index per location."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        if isinstance(component, (bool, np.bool_)) or not isinstance(
            component, Integral
        ):
            raise TypeError("component must be an integer index.")
        component_index = int(component)
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(f"component must lie in [0, {self.n_components - 1}].")
        return np.argmax(np.abs(self.loadings_[:, :, component_index]), axis=1)

    def to_frame(self) -> pd.DataFrame:
        """Return local variance proportions and winning PC1 variable."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        data: dict[str, Any] = {
            f"Comp.{index + 1}_PV": self.local_pv_[:, index]
            for index in range(self.n_components)
        }
        data["local_CP"] = self.cumulative_pv_
        winners = self.get_winning_variable(0)
        data["win_var_PC1"] = np.asarray(self.feature_names_, dtype=object)[winners]
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Return global and local variance diagnostics as a plain-text table."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        winning_pc1 = self.get_winning_variable(0)
        return format_summary(
            "GWPCA Summary",
            {
                "n_components": self.n_components,
                "bandwidth": self.bandwidth_,
                "global_variance": float(
                    np.sum(
                        self.pca_global_.explained_variance_ratio_[: self.n_components]
                    )
                    * 100.0
                ),
                "local_variance_mean": float(np.mean(self.cumulative_pv_)),
                "local_variance_std": float(np.std(self.cumulative_pv_)),
                "local_variance_range": (
                    float(np.min(self.cumulative_pv_)),
                    float(np.max(self.cumulative_pv_)),
                ),
                "winning_var_pc1_mode": int(
                    np.argmax(
                        np.bincount(winning_pc1, minlength=len(self.feature_names_))
                    )
                ),
            },
        )
