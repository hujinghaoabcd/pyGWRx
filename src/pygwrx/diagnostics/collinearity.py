# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Local collinearity diagnostics for fitted GWR models.

This module computes weighted local correlations, variance inflation factors, condition numbers, and variance-decomposition proportions.

Author:
    Jinghao Hu
"""

__author__ = "Jinghao Hu"
__license__ = "MIT"

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class LocalCollinearityDiagnostics:
    """Diagnose spatially varying multicollinearity in a fitted GWR model.

    The diagnostic is intended for fitted, single-bandwidth spatial GWR-like
    estimators.  It uses the model's training coordinates, kernel, fitted
    bandwidth, adaptive/fixed bandwidth setting, distance metric, and
    intercept setting.

    Args:
        gwr_model: Fitted GWR-like model.  The model must expose ``X_train_``,
            ``coords_train_``, ``bandwidth_``, ``kernel``, ``adaptive``,
            ``distance_metric``, ``fit_intercept``, and ``_is_fitted``.
        tolerance: Numerical tolerance used for constant columns, rank checks, and small
            singular values.

    Notes:
        The following quantities are computed at every calibration location:

        * local weighted correlations between predictor pairs;
        * variance inflation factors (VIF), excluding the intercept;
        * condition number (CN), using the actual local design matrix and therefore
          including the intercept when ``fit_intercept=True``;
        * variance decomposition proportions (VDP), with axes ordered as
          ``(location, condition component, design variable)``.

        This implementation does not silently approximate MGWR, GTWR, STWR, SGTWR, or
        other multiscale/spatiotemporal weighting schemes.  Those models require
        model-specific diagnostics.
    """

    def __init__(self, gwr_model: Any, tolerance: float = 1e-10) -> None:
        self.model = gwr_model
        self.tolerance = self._validate_tolerance(tolerance)
        self._cache: Optional[Dict[str, Any]] = None
        self._distance_matrix: Optional[np.ndarray] = None

        self._validate_model()

        self.X_features_ = np.asarray(self.model.X_train_, dtype=float)
        self.coords_ = np.asarray(self.model.coords_train_, dtype=float)
        self.n_samples_, self.n_features_ = self.X_features_.shape

        self.feature_names_ = self._resolve_feature_names()
        self.correlation_pairs_ = list(combinations(range(self.n_features_), 2))
        self.correlation_pair_names_ = [
            (self.feature_names_[left], self.feature_names_[right])
            for left, right in self.correlation_pairs_
        ]

        if bool(getattr(self.model, "fit_intercept", True)):
            self.X_design_ = np.column_stack(
                [np.ones(self.n_samples_, dtype=float), self.X_features_]
            )
            self.design_names_ = ["intercept"] + self.feature_names_
        else:
            self.X_design_ = self.X_features_.copy()
            self.design_names_ = list(self.feature_names_)

    @staticmethod
    def _validate_tolerance(tolerance: float) -> float:
        if isinstance(tolerance, (bool, np.bool_)):
            raise TypeError("tolerance must be a positive finite float.")
        try:
            value = float(tolerance)
        except (TypeError, ValueError) as exc:
            raise TypeError("tolerance must be a positive finite float.") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        return value

    def _validate_model(self) -> None:
        if not bool(getattr(self.model, "_is_fitted", False)):
            raise ValueError(
                "Model must be fitted before running diagnostics. "
                "Call model.fit(X, y, coords) first."
            )

        required = (
            "X_train_",
            "coords_train_",
            "bandwidth_",
            "kernel",
            "adaptive",
            "distance_metric",
            "fit_intercept",
        )
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "Model does not provide the attributes required for local "
                "collinearity diagnostics: {}.".format(", ".join(missing))
            )

        # Explicitly reject weighting structures that this spatial diagnostic
        # cannot reproduce correctly.
        if getattr(self.model, "bandwidths_", None) is not None:
            raise NotImplementedError(
                "LocalCollinearityDiagnostics currently supports one fitted "
                "spatial bandwidth only; multiscale models require "
                "variable-specific diagnostics."
            )
        if getattr(self.model, "times_train_", None) is not None:
            raise NotImplementedError(
                "Spatiotemporal models require diagnostics based on their "
                "combined space-time weights."
            )
        if (
            getattr(self.model, "spatial_bandwidth_", None) is not None
            or getattr(self.model, "temporal_bandwidth_", None) is not None
        ):
            raise NotImplementedError(
                "Separate spatial and temporal bandwidths are not supported by "
                "this spatial-only diagnostic."
            )

        X = np.asarray(self.model.X_train_)
        coords = np.asarray(self.model.coords_train_)

        if X.ndim != 2:
            raise ValueError("model.X_train_ must be a two-dimensional array.")
        if X.shape[1] < 2:
            raise ValueError(
                "Local collinearity diagnostics require at least two predictor "
                "variables."
            )
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("model.coords_train_ must have shape (n_samples, 2).")
        if coords.shape[0] != X.shape[0]:
            raise ValueError(
                "model.X_train_ and model.coords_train_ must contain the same "
                "number of samples."
            )

        try:
            X_float = X.astype(float, copy=False)
            coords_float = coords.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Training features and coordinates must be numeric."
            ) from exc

        if not np.isfinite(X_float).all():
            raise ValueError("model.X_train_ contains NaN or infinite values.")
        if not np.isfinite(coords_float).all():
            raise ValueError("model.coords_train_ contains NaN or infinite values.")

        bandwidth = np.asarray(self.model.bandwidth_)
        if bandwidth.ndim != 0:
            raise NotImplementedError(
                "Only a single scalar fitted bandwidth is supported."
            )
        bandwidth_value = float(bandwidth)
        if not np.isfinite(bandwidth_value) or bandwidth_value <= 0.0:
            raise ValueError("model.bandwidth_ must be positive and finite.")

        if bool(self.model.adaptive):
            k = int(round(bandwidth_value))
            if not np.isclose(bandwidth_value, k):
                raise ValueError(
                    "For adaptive diagnostics, model.bandwidth_ must be an "
                    "integer number of nearest neighbours."
                )
            if k < 1 or k > X.shape[0]:
                raise ValueError(
                    "Adaptive bandwidth k must satisfy 1 <= k <= n_samples."
                )

    def _resolve_feature_names(self) -> List[str]:
        names = getattr(self.model, "feature_names_in_", None)
        if names is None:
            return ["x{}".format(i) for i in range(self.n_features_)]

        values = [str(name) for name in np.asarray(names).tolist()]
        if len(values) != self.n_features_ or len(set(values)) != len(values):
            return ["x{}".format(i) for i in range(self.n_features_)]
        return values

    def _get_distance_matrix(self) -> np.ndarray:
        if self._distance_matrix is None:
            from pygwrx.core.utils import compute_distance_matrix

            self._distance_matrix = compute_distance_matrix(
                self.coords_,
                self.coords_,
                metric=str(self.model.distance_metric),
            )
        return self._distance_matrix

    def _get_local_weights(self, i: int) -> np.ndarray:
        """Return model-consistent kernel weights for calibration location ``i``."""
        if not isinstance(i, (int, np.integer)):
            raise TypeError("i must be an integer location index.")
        if i < 0 or i >= self.n_samples_:
            raise IndexError("Location index is outside the training sample range.")

        from pygwrx.core.kernels import get_kernel_function

        distances = self._get_distance_matrix()[int(i)]
        kernel_func = get_kernel_function(self.model.kernel)
        bandwidth = float(self.model.bandwidth_)

        if bool(self.model.adaptive):
            k = int(round(bandwidth))
            local_bandwidth = float(np.partition(distances, k - 1)[k - 1])
            # Compact kernels use d < h. Move h by one representable value so
            # that the k-th neighbour is included rather than receiving zero.
            local_bandwidth = float(np.nextafter(local_bandwidth, np.inf))
            if local_bandwidth <= 0.0 or not np.isfinite(local_bandwidth):
                raise ValueError(
                    "Unable to derive a positive adaptive bandwidth at "
                    "location {}.".format(i)
                )
        else:
            local_bandwidth = bandwidth

        weights = np.asarray(kernel_func(distances, local_bandwidth), dtype=float)
        if weights.shape != (self.n_samples_,):
            raise ValueError(
                "Kernel function must return one weight per training sample."
            )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("Kernel function returned invalid spatial weights.")

        total = float(np.sum(weights))
        if total <= self.tolerance:
            raise ValueError(
                "Local spatial weights sum to zero at location {}. Increase the "
                "bandwidth or use a kernel with wider support.".format(i)
            )
        return weights / total

    def _weighted_correlation(
        self, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.sum(weights[:, np.newaxis] * self.X_features_, axis=0)
        centered = self.X_features_ - mean
        covariance = centered.T @ (centered * weights[:, np.newaxis])
        variances = np.diag(covariance).copy()
        constant = variances <= self.tolerance

        correlation = np.full((self.n_features_, self.n_features_), np.nan, dtype=float)
        nonconstant = ~constant
        if np.any(nonconstant):
            standard_deviation = np.sqrt(variances[nonconstant])
            sub_covariance = covariance[np.ix_(nonconstant, nonconstant)]
            sub_correlation = sub_covariance / np.outer(
                standard_deviation, standard_deviation
            )
            sub_correlation = np.clip(sub_correlation, -1.0, 1.0)
            np.fill_diagonal(sub_correlation, 1.0)
            correlation[np.ix_(nonconstant, nonconstant)] = sub_correlation

        return correlation, constant

    def _vif_from_correlation(
        self, correlation: np.ndarray, constant: np.ndarray
    ) -> np.ndarray:
        vif = np.full(self.n_features_, np.inf, dtype=float)
        nonconstant_indices = np.flatnonzero(~constant)

        if nonconstant_indices.size == 0:
            return vif
        if nonconstant_indices.size == 1:
            vif[nonconstant_indices[0]] = 1.0
            return vif

        corr_sub = correlation[np.ix_(nonconstant_indices, nonconstant_indices)]
        try:
            eigenvalues = np.linalg.eigvalsh(corr_sub)
        except np.linalg.LinAlgError:
            return vif

        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= self.tolerance:
            return vif

        try:
            inverse = np.linalg.solve(corr_sub, np.eye(corr_sub.shape[0]))
        except np.linalg.LinAlgError:
            return vif

        local_vif = np.diag(inverse)
        local_vif = np.maximum(local_vif, 1.0)
        vif[nonconstant_indices] = local_vif
        return vif

    def _condition_and_vdp(
        self, weights: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        q = self.X_design_.shape[1]
        sqrt_weights = np.sqrt(weights)
        weighted_design = self.X_design_ * sqrt_weights[:, np.newaxis]
        column_norms = np.linalg.norm(weighted_design, axis=0)

        if np.any(column_norms <= self.tolerance):
            return (
                np.inf,
                np.full((q, q), np.nan),
                np.full(q, np.nan),
                np.full(q, np.inf),
            )

        scaled_design = weighted_design / column_norms

        try:
            # full_matrices=True guarantees a complete q x q right-singular
            # vector basis even when the effective local sample is smaller
            # than the number of design columns.
            _, singular_compact, vt = np.linalg.svd(scaled_design, full_matrices=True)
        except np.linalg.LinAlgError:
            return (
                np.inf,
                np.full((q, q), np.nan),
                np.full(q, np.nan),
                np.full(q, np.inf),
            )

        singular_values = np.zeros(q, dtype=float)
        singular_values[: singular_compact.size] = singular_compact
        eigenvectors = vt.T

        largest = singular_values[0]
        if largest <= self.tolerance:
            condition_number = np.inf
            condition_indices = np.full(q, np.inf)
            positive_components = np.zeros(q, dtype=bool)
        else:
            singular_tolerance = max(
                self.tolerance * largest,
                np.finfo(float).eps * max(scaled_design.shape) * largest,
            )
            positive_components = singular_values > singular_tolerance
            condition_indices = np.full(q, np.inf, dtype=float)
            condition_indices[positive_components] = (
                largest / singular_values[positive_components]
            )
            condition_number = float(condition_indices[-1])

        # raw[var, component] = (V[var, component] / s_component)^2
        raw = np.zeros((q, q), dtype=float)
        if np.any(positive_components):
            raw[:, positive_components] = (
                eigenvectors[:, positive_components]
                / singular_values[positive_components][np.newaxis, :]
            ) ** 2

        zero_components = ~positive_components
        proportions = np.full((q, q), np.nan, dtype=float)
        for variable in range(q):
            zero_loading = eigenvectors[variable, zero_components] ** 2
            if zero_loading.size and np.sum(zero_loading) > self.tolerance:
                values = np.zeros(q, dtype=float)
                values[zero_components] = zero_loading / np.sum(zero_loading)
                proportions[variable] = values
            else:
                denominator = float(np.sum(raw[variable]))
                if denominator > self.tolerance:
                    proportions[variable] = raw[variable] / denominator

        # Public order: component x variable.
        return (
            condition_number,
            proportions.T,
            singular_values,
            condition_indices,
        )

    def _compute_all(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache

        n = self.n_samples_
        p = self.n_features_
        q = self.X_design_.shape[1]
        n_pairs = len(self.correlation_pairs_)

        correlations = np.full((n, n_pairs), np.nan, dtype=float)
        vif = np.full((n, p), np.nan, dtype=float)
        condition_number = np.full(n, np.nan, dtype=float)
        vdp = np.full((n, q, q), np.nan, dtype=float)
        singular_values = np.full((n, q), np.nan, dtype=float)
        condition_indices = np.full((n, q), np.nan, dtype=float)
        effective_neighbors = np.zeros(n, dtype=int)

        for i in range(n):
            weights = self._get_local_weights(i)
            effective_neighbors[i] = int(np.count_nonzero(weights > self.tolerance))

            corr_matrix, constant = self._weighted_correlation(weights)
            for pair_index, (left, right) in enumerate(self.correlation_pairs_):
                correlations[i, pair_index] = corr_matrix[left, right]
            vif[i] = self._vif_from_correlation(corr_matrix, constant)

            (
                condition_number[i],
                vdp[i],
                singular_values[i],
                condition_indices[i],
            ) = self._condition_and_vdp(weights)

        self._cache = {
            "local_correlations": correlations,
            "correlation_pairs": list(self.correlation_pair_names_),
            "vif": vif,
            "condition_number": condition_number,
            "vdp": vdp,
            "singular_values": singular_values,
            "condition_indices": condition_indices,
            "feature_names": list(self.feature_names_),
            "design_names": list(self.design_names_),
            "effective_neighbors": effective_neighbors,
        }
        return self._cache

    def compute_local_correlations(self) -> np.ndarray:
        """Return local weighted correlations for every predictor pair.

        Returns:
            ndarray of shape (n_samples, n_feature_pairs): Pair order is available from ``correlation_pair_names_`` and from
                ``diagnose()['correlation_pairs']``.  Correlations involving a
                locally constant predictor are reported as ``NaN``.
        """
        return self._compute_all()["local_correlations"].copy()

    def compute_vif(self) -> np.ndarray:
        """Return local VIF values with shape ``(n_samples, n_features)``.

        The intercept is excluded.  Locally constant predictors and singular
        local correlation matrices are represented by ``np.inf`` rather than
        by deceptively finite pseudo-inverse values.
        """
        return self._compute_all()["vif"].copy()

    def compute_condition_number(self) -> np.ndarray:
        """Return the scaled local design-matrix condition number."""
        return self._compute_all()["condition_number"].copy()

    def compute_vdp(self) -> np.ndarray:
        """Return variance decomposition proportions.

        Returns:
            ndarray of shape (n_samples, n_design_variables, n_design_variables): ``vdp[i, j, k]`` is the share of design variable ``k``'s variance
                associated with condition component ``j`` at location ``i``.
                When an intercept is fitted, it is included as the first design
                variable and named ``"intercept"``.
        """
        return self._compute_all()["vdp"].copy()

    @staticmethod
    def _max_preserving_infinity(values: np.ndarray) -> float:
        if np.isposinf(values).any():
            return np.inf
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else np.nan

    @staticmethod
    def _mean_preserving_infinity(values: np.ndarray) -> float:
        if np.isposinf(values).any():
            return np.inf
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else np.nan

    def diagnose(self, verbose: bool = True) -> Dict[str, Any]:
        """Run and summarize the complete local collinearity diagnostic."""
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be a boolean.")

        computed = self._compute_all()
        vif = computed["vif"]
        cn = computed["condition_number"]

        severe_vif_cells = np.where(vif > 10.0)
        severe_vif_location_mask = np.any(vif > 10.0, axis=1)
        severe_cn_location_mask = cn > 30.0

        finite_cn = cn[np.isfinite(cn)]
        summary = {
            "max_vif": self._max_preserving_infinity(vif),
            "mean_vif": self._mean_preserving_infinity(vif),
            "max_cn": self._max_preserving_infinity(cn),
            "median_cn": (
                np.inf
                if np.isposinf(cn).any()
                else float(np.median(finite_cn)) if finite_cn.size else np.nan
            ),
            "pct_severe_vif_locations": float(
                np.mean(severe_vif_location_mask) * 100.0
            ),
            "pct_severe_vif_cells": float(np.mean(vif > 10.0) * 100.0),
            "pct_severe_cn_locations": float(np.mean(severe_cn_location_mask) * 100.0),
            "n_infinite_vif_locations": int(np.sum(np.any(np.isposinf(vif), axis=1))),
            "n_infinite_cn_locations": int(np.sum(np.isposinf(cn))),
        }
        # Backwards-compatible aliases, now with correct location semantics.
        summary["pct_severe_vif"] = summary["pct_severe_vif_locations"]
        summary["pct_severe_cn"] = summary["pct_severe_cn_locations"]

        diagnostics: Dict[str, Any] = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in computed.items()
        }
        diagnostics["severe_multicollinearity"] = {
            "vif_locations": np.flatnonzero(severe_vif_location_mask),
            "vif_cells": severe_vif_cells,
            "cn_locations": np.flatnonzero(severe_cn_location_mask),
        }
        diagnostics["summary"] = summary

        if verbose:
            self._print_summary(summary)

        return diagnostics

    def to_frame(self) -> pd.DataFrame:
        """Return row-wise collinearity diagnostics as a tidy table."""
        computed = self._compute_all()
        data: Dict[str, Any] = {
            "coord_0": self.coords_[:, 0],
            "coord_1": self.coords_[:, 1],
            "condition_number": computed["condition_number"],
            "effective_neighbors": computed["effective_neighbors"],
        }
        for index, name in enumerate(self.feature_names_):
            data[f"vif_{name}"] = computed["vif"][:, index]
        for index, (left, right) in enumerate(self.correlation_pair_names_):
            data[f"corr_{left}__{right}"] = computed["local_correlations"][:, index]
        return pd.DataFrame(data)

    def summary_frame(self) -> pd.DataFrame:
        """Return the diagnostic summary as a one-row table."""
        return pd.DataFrame([self.diagnose(verbose=False)["summary"]])

    @staticmethod
    def _format_number(value: float) -> str:
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        return "{:.2f}".format(value)

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print("Local Collinearity Diagnostics Summary")
        print("=" * 70)
        print("\nVIF Statistics:")
        print("  Maximum VIF: {}".format(self._format_number(summary["max_vif"])))
        print("  Mean VIF: {}".format(self._format_number(summary["mean_vif"])))
        print(
            "  Locations with VIF > 10: {:.1f}%".format(
                summary["pct_severe_vif_locations"]
            )
        )
        print("\nCondition Number Statistics:")
        print("  Maximum CN: {}".format(self._format_number(summary["max_cn"])))
        print("  Median CN: {}".format(self._format_number(summary["median_cn"])))
        print(
            "  Locations with CN > 30: {:.1f}%".format(
                summary["pct_severe_cn_locations"]
            )
        )

        max_vif = summary["max_vif"]
        max_cn = summary["max_cn"]
        print("\nInterpretation:")
        if np.isfinite(max_vif) and np.isfinite(max_cn) and max_vif < 5 and max_cn < 30:
            print("  [OK] No significant multicollinearity detected")
        elif (
            np.isfinite(max_vif)
            and np.isfinite(max_cn)
            and max_vif < 10
            and max_cn < 100
        ):
            print("  [WARN] Moderate multicollinearity detected")
            print("         Review the affected variables and locations")
        else:
            print("  [!] Severe or singular local collinearity detected")
            print("      Coefficient estimates may be unstable")
            print("      Consider removing redundant variables, increasing")
            print("      the bandwidth, or using a locally compensated ridge model")
        print("=" * 70)
