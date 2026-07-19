# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Mixed geographically weighted regression.

The estimator exposes one public :class:`MixedGWR` API. Numerical partial-
regression routines are kept in the private ``_mixed_gwr_core`` module.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from pygwrx.core.base import BaseGWR
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.utils import (
    add_intercept,
    compute_distance_matrix,
    validate_coords,
    validate_data,
)
from pygwrx.models._mixed_gwr_core import (
    compute_mixed_gwr_hat_matrix,
    fit_mixed_gwr_core,
)

VariableSpec = Optional[Sequence[Union[int, str]]]


class MixedGWR(BaseGWR):
    """Fit a semiparametric GWR with global and local coefficients.

    Mixed GWR partitions explanatory variables into globally constant and
    geographically varying groups. The implementation follows the partial-
    regression algorithm used by ``GWmodel::gwr.mixed`` rather than an
    iterative backfitting algorithm.

    Args:
        kernel: Spatial kernel name or callable.
        bandwidth: Fixed distance, adaptive neighbour count, or one of
            ``"cv"``, ``"aic"``, ``"aicc"``, or ``"bic"``. Automatic
            selection uses the corresponding full-GWR bandwidth as the mixed
            model bandwidth, matching the published Dublin workflow.
        bandwidth_method: Selection method used when ``bandwidth=None`` or the
            legacy token ``"adaptive"`` is supplied.
        adaptive: Whether numeric bandwidth is a neighbour count.
        local_vars: Feature indices or DataFrame column names with local
            coefficients. If omitted with ``global_vars``, all features are
            local.
        global_vars: Feature indices or names with global coefficients. If only
            one variable group is supplied, the other is its complement.
        intercept_fixed: Whether the fitted intercept is global. If false, the
            intercept varies locally.
        ridge: Non-negative regularization applied explicitly to local and global
            normal equations. The default ``0.0`` reproduces the unregularized
            reference algorithm, with deterministic pseudo-inverse fallback.
        fit_intercept: Whether to fit an intercept.
        bandwidth_range: Optional search range for automatic bandwidth selection.
        optimization_method: Bandwidth optimizer passed to the shared selector.
        distance_metric: Distance metric used by the shared distance utility.
        verbose: Whether to print fit progress.

    Attributes:
        coef_local_: Local coefficients with shape ``(n_samples, n_local_vars)``.
        coef_global_: Constant coefficients for the global feature variables.
        intercept_: Scalar global intercept or a vector of local intercepts.
        coef_: Full coefficient surface in original feature order.
        enp_: Effective parameter count ``trace(S)`` when diagnostics are enabled.

    References:
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically Weighted Regression. Wiley.
    """

    def __init__(
        self,
        kernel: Union[str, Callable] = "bisquare",
        bandwidth: Union[float, int, str, None] = "aicc",
        bandwidth_method: str = "aicc",
        adaptive: bool = True,
        local_vars: VariableSpec = None,
        global_vars: VariableSpec = None,
        intercept_fixed: bool = True,
        ridge: float = 0.0,
        fit_intercept: bool = True,
        bandwidth_range: Optional[tuple[float, float]] = None,
        optimization_method: str = "golden_section",
        distance_metric: str = "euclidean",
        verbose: bool = False,
    ) -> None:
        super().__init__(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method=bandwidth_method,
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
            verbose=verbose,
        )
        if not isinstance(intercept_fixed, (bool, np.bool_)):
            raise TypeError("intercept_fixed must be boolean.")
        if isinstance(ridge, (bool, np.bool_)):
            raise TypeError("ridge must be a real scalar, not bool.")
        ridge_value = float(ridge)
        if not np.isfinite(ridge_value) or ridge_value < 0:
            raise ValueError("ridge must be finite and non-negative.")

        self.local_vars = local_vars
        self.global_vars = global_vars
        self.intercept_fixed = bool(intercept_fixed)
        self.ridge = ridge_value
        self._reset_mixed_state()

    def _reset_mixed_state(self) -> None:
        """Clear fitted Mixed GWR state."""
        self._mark_unfitted()
        self._reset_regression_state()
        self._reset_gwr_state()
        self.coef_local_: Optional[np.ndarray] = None
        self.coef_global_: Optional[np.ndarray] = None
        self.local_var_indices_: Optional[np.ndarray] = None
        self.global_var_indices_: Optional[np.ndarray] = None
        self.enp_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.aicc_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.trace_StS_: Optional[float] = None
        self.selection_history_ = None
        self._coef_local_design_: Optional[np.ndarray] = None
        self._coef_global_design_: Optional[np.ndarray] = None
        self._X_local_design_train_: Optional[np.ndarray] = None
        self._X_global_design_train_: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_variable_spec(
        values: VariableSpec,
        *,
        name: str,
        n_features: int,
        feature_names: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Convert feature names or indices to validated unique integer indices."""
        if values is None:
            return None
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a sequence of feature names or indices.")
        values_list = list(values)
        if not values_list:
            return np.empty(0, dtype=int)
        uses_names = all(isinstance(value, str) for value in values_list)
        uses_indices = all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in values_list
        )
        if not uses_names and not uses_indices:
            raise TypeError(
                f"{name} must contain only strings or only integer indices."
            )
        if uses_names:
            if feature_names is None:
                raise ValueError(f"{name} uses names, but X is not a DataFrame.")
            lookup = {str(value): index for index, value in enumerate(feature_names)}
            missing = [value for value in values_list if value not in lookup]
            if missing:
                raise ValueError(f"Unknown feature names in {name}: {missing}.")
            indices = np.asarray([lookup[value] for value in values_list], dtype=int)
        else:
            indices = np.asarray(values_list, dtype=int)
            if np.any(indices < 0) or np.any(indices >= n_features):
                raise ValueError(f"{name} contains an out-of-range feature index.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} contains duplicate variables.")
        return indices

    def _partition_variables(self, n_features: int) -> None:
        """Resolve local and global feature groups in original feature space."""
        feature_names = self.feature_names_in_
        local = self._normalize_variable_spec(
            self.local_vars,
            name="local_vars",
            n_features=n_features,
            feature_names=feature_names,
        )
        global_ = self._normalize_variable_spec(
            self.global_vars,
            name="global_vars",
            n_features=n_features,
            feature_names=feature_names,
        )
        all_indices = np.arange(n_features, dtype=int)

        if local is None and global_ is None:
            local = all_indices
            global_ = np.empty(0, dtype=int)
        elif local is None:
            assert global_ is not None
            local = np.setdiff1d(all_indices, global_, assume_unique=True)
        elif global_ is None:
            global_ = np.setdiff1d(all_indices, local, assume_unique=True)
        else:
            overlap = np.intersect1d(local, global_)
            if overlap.size:
                raise ValueError(
                    f"local_vars and global_vars overlap at indices {overlap.tolist()}."
                )
            covered = np.sort(np.concatenate([local, global_]))
            if not np.array_equal(covered, all_indices):
                missing = np.setdiff1d(all_indices, covered)
                raise ValueError(
                    "When both variable groups are supplied, they must partition all "
                    f"features; missing indices {missing.tolist()}."
                )

        if local.size == 0:
            raise ValueError("MixedGWR requires at least one local feature variable.")
        self.local_var_indices_ = np.asarray(local, dtype=int)
        self.global_var_indices_ = np.asarray(global_, dtype=int)

    def _build_designs(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build local and global design matrices, including the chosen intercept."""
        if self.local_var_indices_ is None or self.global_var_indices_ is None:
            raise RuntimeError("Variable groups have not been initialized.")
        X_local = X[:, self.local_var_indices_]
        X_global = X[:, self.global_var_indices_]
        if self.fit_intercept:
            if self.intercept_fixed:
                X_global = add_intercept(X_global)
            else:
                X_local = add_intercept(X_local)
        return X_local, X_global

    def _resolve_mixed_bandwidth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
    ) -> Union[int, float]:
        """Resolve a numeric bandwidth using the shared GWR selector when needed."""
        from pygwrx.core.bandwidth import get_bandwidth_selector

        full_design = add_intercept(X) if self.fit_intercept else X
        if isinstance(self.bandwidth, str) or self.bandwidth is None:
            token = (
                self.bandwidth.strip().lower()
                if isinstance(self.bandwidth, str)
                else self.bandwidth_method
            )
            if token == "adaptive":
                token = self.bandwidth_method
            if token not in {"cv", "aic", "aicc", "bic"}:
                raise ValueError(
                    "Automatic MixedGWR bandwidth must use 'cv', 'aic', "
                    "'aicc', or 'bic'."
                )
            selector = get_bandwidth_selector(
                token,
                adaptive=self.adaptive,
                verbose=self.verbose,
                optimization_method=self.optimization_method,
            )
            selected = selector.select(
                full_design,
                y,
                coords,
                self.kernel_func_,
                bandwidth_range=self.bandwidth_range,
                distance_metric=self.distance_metric,
            )
            return int(selected) if self.adaptive else float(selected)

        value = float(self.bandwidth)
        if self.adaptive:
            if not value.is_integer():
                raise ValueError(
                    "Adaptive MixedGWR bandwidth must be an integer neighbour count."
                )
            k = int(value)
            minimum = full_design.shape[1] + 1
            if k < minimum or k > X.shape[0]:
                raise ValueError(
                    f"Adaptive bandwidth must satisfy {minimum} <= k <= {X.shape[0]}."
                )
            return k
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Fixed bandwidth must be finite and greater than zero.")
        return value

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
        compute_enp: bool = True,
    ) -> "MixedGWR":
        """Fit the mixed global/local coefficient model."""
        self._reset_mixed_state()
        try:
            if not isinstance(compute_enp, (bool, np.bool_)):
                raise TypeError("compute_enp must be boolean.")

            X_arr, y_arr = validate_data(X, y)
            coords_arr = validate_coords(coords)
            if X_arr.shape[0] != coords_arr.shape[0]:
                raise ValueError(
                    "X, y, and coords must contain the same number of rows."
                )
            self.n_samples_ = X_arr.shape[0]
            self.n_features_in_ = X_arr.shape[1]
            self.feature_names_in_ = (
                np.asarray(X.columns, dtype=object)
                if isinstance(X, pd.DataFrame)
                else None
            )
            self._partition_variables(X_arr.shape[1])

            self.X_train_ = X_arr.copy()
            self.y_train_ = y_arr.copy()
            self.coords_train_ = coords_arr.copy()
            self.kernel_func_ = get_kernel_function(self.kernel)
            self.bandwidth_ = self._resolve_mixed_bandwidth(X_arr, y_arr, coords_arr)

            X_local, X_global = self._build_designs(X_arr)
            distances = compute_distance_matrix(
                coords_arr,
                coords_arr,
                metric=self.distance_metric,
            )
            local_design_coef, global_design_coef = fit_mixed_gwr_core(
                X_local,
                X_global,
                y_arr,
                self.bandwidth_,
                self.kernel_func_,
                distances,
                adaptive=self.adaptive,
                ridge=self.ridge,
            )
            self._coef_local_design_ = local_design_coef.copy()
            self._coef_global_design_ = global_design_coef.copy()
            self._X_local_design_train_ = X_local.copy()
            self._X_global_design_train_ = X_global.copy()

            self.fitted_values_ = np.einsum("ij,ij->i", X_local, local_design_coef) + (
                X_global @ global_design_coef if X_global.shape[1] else 0.0
            )
            self.residuals_ = y_arr - self.fitted_values_

            if self.fit_intercept and self.intercept_fixed:
                self.intercept_ = float(global_design_coef[0])
                self.coef_global_ = global_design_coef[1:].copy()
                self.coef_local_ = local_design_coef.copy()
            elif self.fit_intercept:
                self.intercept_ = local_design_coef[:, 0].copy()
                self.coef_local_ = local_design_coef[:, 1:].copy()
                self.coef_global_ = global_design_coef.copy()
            else:
                self.intercept_ = 0.0
                self.coef_local_ = local_design_coef.copy()
                self.coef_global_ = global_design_coef.copy()

            self.coef_ = np.empty((X_arr.shape[0], X_arr.shape[1]), dtype=float)
            self.coef_[:, self.local_var_indices_] = self.coef_local_
            if self.global_var_indices_.size:
                self.coef_[:, self.global_var_indices_] = self.coef_global_[None, :]

            if compute_enp:
                self.hat_matrix_ = compute_mixed_gwr_hat_matrix(
                    X_local,
                    X_global,
                    self.bandwidth_,
                    self.kernel_func_,
                    distances,
                    adaptive=self.adaptive,
                    ridge=self.ridge,
                )
                self.enp_ = float(np.trace(self.hat_matrix_))
                self.trace_StS_ = float(np.sum(self.hat_matrix_**2))
                self.diagnostics_ = compute_diagnostics(
                    y_arr,
                    self.fitted_values_,
                    hat_matrix=self.hat_matrix_,
                    compute_gwr_stats=True,
                )
            else:
                self.hat_matrix_ = None
                self.enp_ = None
                self.trace_StS_ = None
                self.diagnostics_ = compute_diagnostics(
                    y_arr,
                    self.fitted_values_,
                    n_features=X_local.shape[1] + X_global.shape[1],
                )

            self.aic_ = float(self.diagnostics_["aic"])
            self.aicc_ = float(self.diagnostics_["aicc"])
            self.bic_ = float(self.diagnostics_["bic"])
            self._mark_fitted()
            return self
        except Exception:
            self._reset_mixed_state()
            raise

    def _validate_prediction_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate prediction features, feature order, and coordinates."""
        self._check_is_fitted()
        if isinstance(X, pd.DataFrame) and self.feature_names_in_ is not None:
            received = np.asarray(X.columns, dtype=object)
            if not np.array_equal(received, self.feature_names_in_):
                raise ValueError(
                    "Prediction DataFrame columns must match training columns in the "
                    f"same order. Expected {self.feature_names_in_.tolist()}, "
                    f"got {received.tolist()}."
                )
        X_arr, _ = validate_data(X, np.zeros(len(X), dtype=float))
        coords_arr = validate_coords(coords)
        if X_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but the fitted model expects "
                f"{self.n_features_in_}."
            )
        return X_arr, coords_arr

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Predict at new locations without modifying fitted training state."""
        X_arr, coords_arr = self._validate_prediction_inputs(X, coords)
        if (
            self.X_train_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self._coef_global_design_ is None
            or self._X_local_design_train_ is None
            or self._X_global_design_train_ is None
            or self.kernel_func_ is None
            or self.bandwidth_ is None
        ):
            raise RuntimeError("Stored MixedGWR training state is incomplete.")

        X_local_test, X_global_test = self._build_designs(X_arr)
        target_distances = compute_distance_matrix(
            coords_arr,
            self.coords_train_,
            metric=self.distance_metric,
        )
        training_distances = compute_distance_matrix(
            self.coords_train_,
            self.coords_train_,
            metric=self.distance_metric,
        )
        local_coefficients, global_coefficients = fit_mixed_gwr_core(
            self._X_local_design_train_,
            self._X_global_design_train_,
            self.y_train_,
            self.bandwidth_,
            self.kernel_func_,
            training_distances,
            target_distances=target_distances,
            adaptive=self.adaptive,
            ridge=self.ridge,
        )
        # Re-estimation is deterministic and should reproduce the stored global vector.
        if not np.allclose(
            global_coefficients, self._coef_global_design_, rtol=1e-10, atol=1e-10
        ):
            raise RuntimeError(
                "Stored and recomputed global coefficients are inconsistent."
            )
        return np.einsum("ij,ij->i", X_local_test, local_coefficients) + (
            X_global_test @ global_coefficients if X_global_test.shape[1] else 0.0
        )

    def test_spatial_variation(self) -> dict:
        """Return descriptive variation of fitted local coefficients.

        This method is descriptive and is not a formal hypothesis test.
        """
        self._check_is_fitted()
        return {
            "local_var_indices": self.local_var_indices_.copy(),
            "global_var_indices": self.global_var_indices_.copy(),
            "local_coef_variance": np.var(self.coef_local_, axis=0),
            "coef_global": self.coef_global_.copy(),
        }
