# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Latent-geometry geographically weighted regression.

LG-GWR learns an interpretable linear distance geometry from coordinates and
context attributes, then performs Gaussian local weighted least squares in the
learned geometry.  The implementation is NumPy-only and uses an analytical
leave-one-out gradient.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from pygwrx.core._summary import format_summary
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.utils import add_intercept, validate_coords

ArrayLike = Union[np.ndarray, pd.DataFrame]
VectorLike = Union[np.ndarray, pd.Series]
BandwidthLike = Union[float, int, Tuple[float, float], None]


@dataclass(frozen=True)
class LGGWRPredictionResult:
    """Detailed LG-GWR predictions at evaluation locations."""

    predictions: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    coords: np.ndarray
    latent_coords: np.ndarray
    feature_names: Tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        """Return predictions and local parameters as a DataFrame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "prediction": self.predictions,
            "intercept": self.intercepts,
        }
        for index in range(self.latent_coords.shape[1]):
            data[f"latent_{index}"] = self.latent_coords[:, index]
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coefficients[:, index]
        return pd.DataFrame(data)


@dataclass(frozen=True)
class _OptimisationResult:
    matrix: np.ndarray
    loss_history: Tuple[float, ...]
    best_loss: float
    final_loss: float
    n_iter: int
    converged: bool
    stop_reason: str


class LGGWR:
    r"""Latent-Geometry Geographically Weighted Regression.

    For observation input :math:`u_i=[s_i,a_i]`, joint LG-GWR learns a linear
    map :math:`z_i=A u_i` and defines

    .. math::

        w_{ij}=K(\|z_i-z_j\|/h).

    The map is trained against leave-one-out prediction error.  The default
    Frobenius-norm constraint fixes the otherwise unidentified global scale of
    ``A``; the bandwidth carries that scale.  Consequently, ordinary L2
    regularisation is allowed only when ``scale_constraint="none"``.

    The separable form keeps geographic distance as one channel and learns an
    attribute map :math:`\zeta_i=B a_i` for a second multiplicative channel,

    .. math::

        w_{ij}=K(d_{ij}^{geo}/h_g)K(\|\zeta_i-\zeta_j\|/h_a).

    With :math:`h_a=\infty`, the separable model reduces exactly to geographic
    GWR at the same geographic bandwidth.

    Args:
        latent_dim: Dimension of the learned latent space.
        bandwidth: Joint latent bandwidth.  In separable mode, a two-item tuple
            supplies ``(h_g, h_a)``; a scalar supplies ``h_g`` and leaves
            ``h_a`` automatic.
        adaptive: Interpret a numeric joint bandwidth as a neighbour count and
            convert it once to a fixed latent distance.  The analytical gradient
            itself is for a fixed distance bandwidth.
        kernel: ``"gaussian"``, ``"bisquare"`` or ``"exponential"``.
        geometry: ``"joint"`` or ``"separable"``.
        fit_intercept: Add an unpenalised local intercept.  A legacy leading
            all-ones column is detected and removed before the intercept is added.
        standardize_geometry: Centre coordinates and scale them by one common
            factor (preserving geographic shape), and z-standardise attributes.
        initialization: ``"coordinate"``, ``"random"`` or ``"pca"``.
            Coordinate initialisation makes ordinary geographic geometry the
            first joint candidate.
        n_restarts: Number of deterministic restarts.  The first uses the
            requested initialisation; later restarts are random.
        learning_rate: NumPy Adam learning rate.
        max_iter: Maximum iterations per geometry/bandwidth stage.
        tol: Improvement and convergence tolerance.
        lambda_reg: Frobenius L2 regularisation.  It must be zero while a norm
            or orthogonality constraint is active because the norm is then fixed.
        scale_constraint: ``"frobenius"`` (default), ``"orthogonal"`` or
            ``"none"``.
        orthogonal_constraint: Deprecated compatibility switch.  ``True`` maps
            to ``scale_constraint="orthogonal"``.
        grad_clip: Global gradient-norm clipping threshold.
        patience: Early-stopping patience.
        select_bandwidth: Select reporting bandwidth(s) by AICc.
        bandwidth_updates: Number of additional geometry-training stages after
            AICc bandwidth reselection.  A value of one implements
            geometry -> bandwidth -> geometry -> bandwidth.
        random_state: Reproducibility seed.
        verbose: Print optimisation progress.
    """

    _KERNELS = {"gaussian", "bisquare", "exponential"}
    _GEOMETRIES = {"joint", "separable"}
    _INITIALISATIONS = {"coordinate", "random", "pca"}
    _SCALE_CONSTRAINTS = {"frobenius", "orthogonal", "none"}

    def __init__(
        self,
        latent_dim: int = 2,
        bandwidth: BandwidthLike = None,
        adaptive: bool = False,
        kernel: str = "gaussian",
        geometry: str = "joint",
        learning_rate: float = 0.05,
        max_iter: int = 100,
        tol: float = 1e-6,
        lambda_reg: float = 0.0,
        orthogonal_constraint: Optional[bool] = None,
        grad_clip: float = 10.0,
        patience: int = 20,
        select_bandwidth: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
        *,
        fit_intercept: bool = True,
        standardize_geometry: bool = True,
        initialization: str = "coordinate",
        n_restarts: int = 1,
        scale_constraint: str = "frobenius",
        bandwidth_updates: int = 1,
    ) -> None:
        self.latent_dim = self._positive_int(latent_dim, "latent_dim")
        self.bandwidth = bandwidth
        self.adaptive = self._boolean(adaptive, "adaptive")
        self.kernel = self._choice(kernel, "kernel", self._KERNELS)
        self.geometry = self._choice(geometry, "geometry", self._GEOMETRIES)
        self.learning_rate = self._nonnegative_float(learning_rate, "learning_rate")
        self.max_iter = self._nonnegative_int(max_iter, "max_iter")
        self.tol = self._positive_float(tol, "tol")
        self.lambda_reg = self._nonnegative_float(lambda_reg, "lambda_reg")
        self.grad_clip = self._positive_float(grad_clip, "grad_clip")
        self.patience = self._positive_int(patience, "patience")
        self.select_bandwidth = self._boolean(select_bandwidth, "select_bandwidth")
        self.random_state = random_state
        self.verbose = self._boolean(verbose, "verbose")
        self.fit_intercept = self._boolean(fit_intercept, "fit_intercept")
        self.standardize_geometry = self._boolean(
            standardize_geometry, "standardize_geometry"
        )
        self.initialization = self._choice(
            initialization, "initialization", self._INITIALISATIONS
        )
        self.n_restarts = self._positive_int(n_restarts, "n_restarts")
        self.bandwidth_updates = self._nonnegative_int(
            bandwidth_updates, "bandwidth_updates"
        )

        if orthogonal_constraint is not None:
            orthogonal = self._boolean(orthogonal_constraint, "orthogonal_constraint")
            if orthogonal:
                scale_constraint = "orthogonal"
            warnings.warn(
                "orthogonal_constraint is deprecated; use scale_constraint instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.scale_constraint = self._choice(
            scale_constraint, "scale_constraint", self._SCALE_CONSTRAINTS
        )
        # Compatibility attribute retained for existing user code.
        self.orthogonal_constraint = self.scale_constraint == "orthogonal"

        if self.lambda_reg > 0.0 and self.scale_constraint != "none":
            raise ValueError(
                "lambda_reg must be 0 when scale_constraint fixes the matrix norm. "
                "Use scale_constraint='none' for ordinary L2 regularisation."
            )
        if self.scale_constraint == "none" and self.lambda_reg == 0.0:
            warnings.warn(
                "An unconstrained latent map without regularisation has an "
                "unidentified scale; consider scale_constraint='frobenius'.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._validate_bandwidth_spec()
        self._reset_fit_state()

    # ------------------------------------------------------------------
    # Validation and state
    # ------------------------------------------------------------------
    @staticmethod
    def _boolean(value: bool, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
        return bool(value)

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer.")
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return result

    @staticmethod
    def _nonnegative_int(value: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a non-negative integer.")
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be non-negative.")
        return result

    @staticmethod
    def _positive_float(value: float, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a positive real scalar.")
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero.")
        return result

    @staticmethod
    def _nonnegative_float(value: float, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a non-negative real scalar.")
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return result

    @staticmethod
    def _choice(value: str, name: str, choices: set[str]) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        result = value.strip().lower()
        if result not in choices:
            options = ", ".join(sorted(choices))
            raise ValueError(f"{name} must be one of: {options}.")
        return result

    def _validate_bandwidth_spec(self) -> None:
        value = self.bandwidth
        if value is None:
            return
        if isinstance(value, tuple):
            if self.geometry != "separable" or len(value) != 2:
                raise ValueError(
                    "A bandwidth tuple is supported only in separable mode and must "
                    "contain (h_g, h_a)."
                )
            for item in value:
                if np.isinf(item):
                    continue
                self._positive_float(item, "bandwidth component")
            return
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(
                "bandwidth must be a positive scalar, a two-item tuple, or None."
            )
        if self.adaptive:
            self._positive_int(value, "adaptive bandwidth")
        else:
            self._positive_float(value, "bandwidth")

    def _reset_fit_state(self) -> None:
        self.A_: Optional[np.ndarray] = None
        self.B_: Optional[np.ndarray] = None
        self.metric_matrix_: Optional[np.ndarray] = None
        self.metric_contributions_: Optional[np.ndarray] = None
        self.bandwidth_: Optional[Union[float, Tuple[float, float]]] = None
        self.bandwidth_history_: list[Any] = []
        self.restart_scores_: list[Dict[str, float]] = []
        self.latent_coords_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.hat_matrix_: Optional[np.ndarray] = None
        self.diagnostics_: Optional[Dict[str, float]] = None
        self.loss_history_: list[float] = []
        self.best_loss_: Optional[float] = None
        self.final_loo_loss_: Optional[float] = None
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.stop_reason_: Optional[str] = None
        self.X_train_: Optional[np.ndarray] = None
        self.X_design_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.coords_train_: Optional[np.ndarray] = None
        self.attrs_train_: Optional[np.ndarray] = None
        self.coords_geometry_: Optional[np.ndarray] = None
        self.attrs_geometry_: Optional[np.ndarray] = None
        self.u_train_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.feature_names_: Tuple[str, ...] = ()
        self.geometry_feature_names_: Tuple[str, ...] = ()
        self.n_features_in_: Optional[int] = None
        self.coord_center_: Optional[np.ndarray] = None
        self.coord_scale_: Optional[float] = None
        self.attr_center_: Optional[np.ndarray] = None
        self.attr_scale_: Optional[np.ndarray] = None
        self.constant_attribute_mask_: Optional[np.ndarray] = None
        self._legacy_intercept_input_: bool = False
        self._is_fitted = False

    @staticmethod
    def _numeric_2d(value: Any, name: str) -> np.ndarray:
        raw = value.to_numpy() if isinstance(value, pd.DataFrame) else value
        try:
            array = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values.") from exc
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty two-dimensional array.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return array

    @staticmethod
    def _numeric_y(value: Any) -> np.ndarray:
        raw = value.to_numpy() if isinstance(value, pd.Series) else value
        try:
            array = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("y must contain numeric values.") from exc
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("y must be a non-empty one-dimensional vector.")
        if not np.all(np.isfinite(array)):
            raise ValueError("y contains NaN or infinite values.")
        return array

    def _coerce_X_fit(self, X: ArrayLike) -> Tuple[np.ndarray, Tuple[str, ...]]:
        array = self._numeric_2d(X, "X")
        if isinstance(X, pd.DataFrame):
            names = tuple(str(column) for column in X.columns)
        else:
            names = tuple(f"x{index}" for index in range(array.shape[1]))

        self._legacy_intercept_input_ = False
        if self.fit_intercept and array.shape[1] > 0 and np.allclose(array[:, 0], 1.0):
            self._legacy_intercept_input_ = True
            array = array[:, 1:]
            names = names[1:]
            if array.shape[1] == 0:
                raise ValueError("X must contain at least one non-intercept predictor.")
            warnings.warn(
                "A leading all-ones column was detected and removed because "
                "fit_intercept=True.",
                UserWarning,
                stacklevel=3,
            )
        return array, names

    def _coerce_X_predict(self, X: ArrayLike) -> np.ndarray:
        array = self._numeric_2d(X, "X")
        names = (
            tuple(str(column) for column in X.columns)
            if isinstance(X, pd.DataFrame)
            else None
        )
        if self.fit_intercept and array.shape[1] == (self.n_features_in_ or 0) + 1:
            if np.allclose(array[:, 0], 1.0):
                array = array[:, 1:]
                if names is not None:
                    names = names[1:]
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must contain {self.n_features_in_} predictors; "
                f"got {array.shape[1]}."
            )
        if names is not None and tuple(names) != self.feature_names_:
            raise ValueError(
                "Prediction DataFrame columns must match training columns in the same "
                f"order. Expected {list(self.feature_names_)}, got {list(names)}."
            )
        return array

    @staticmethod
    def _input_names(value: Any, prefix: str, n_columns: int) -> Tuple[str, ...]:
        if isinstance(value, pd.DataFrame):
            return tuple(str(column) for column in value.columns)
        return tuple(f"{prefix}_{index}" for index in range(n_columns))

    def _fit_geometry_scaler(
        self, coords: np.ndarray, attrs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.coord_center_ = np.mean(coords, axis=0)
        centered_coords = coords - self.coord_center_
        axis_scale = np.std(centered_coords, axis=0, ddof=0)
        positive = axis_scale[axis_scale > np.finfo(float).eps]
        self.coord_scale_ = (
            float(np.sqrt(np.mean(positive**2))) if positive.size else 1.0
        )

        self.attr_center_ = np.mean(attrs, axis=0) if attrs.shape[1] else np.zeros(0)
        self.attr_scale_ = (
            np.std(attrs, axis=0, ddof=0) if attrs.shape[1] else np.zeros(0)
        )
        self.constant_attribute_mask_ = (
            self.attr_scale_ <= np.finfo(float).eps
            if attrs.shape[1]
            else np.zeros(0, dtype=bool)
        )
        if attrs.shape[1]:
            self.attr_scale_ = self.attr_scale_.copy()
            self.attr_scale_[self.constant_attribute_mask_] = 1.0
        return self._transform_geometry(coords, attrs)

    def _transform_geometry(
        self, coords: np.ndarray, attrs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.coord_center_ is None or self.coord_scale_ is None:
            raise RuntimeError("Geometry scaler is not fitted.")
        if self.attr_center_ is None or self.attr_scale_ is None:
            raise RuntimeError("Attribute scaler is not fitted.")
        if coords.shape[1] != self.coord_center_.shape[0]:
            raise ValueError("Prediction coordinates have the wrong dimension.")
        if attrs.shape[1] != self.attr_center_.shape[0]:
            raise ValueError("Prediction attributes have the wrong dimension.")
        if not self.standardize_geometry:
            return coords.copy(), attrs.copy()
        coords_scaled = (coords - self.coord_center_) / self.coord_scale_
        attrs_scaled = (
            (attrs - self.attr_center_) / self.attr_scale_
            if attrs.shape[1]
            else attrs.copy()
        )
        return coords_scaled, attrs_scaled

    def _prepare_fit_inputs(
        self,
        X: ArrayLike,
        y: VectorLike,
        coords: ArrayLike,
        attributes: Optional[ArrayLike],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_raw, feature_names = self._coerce_X_fit(X)
        y_arr = self._numeric_y(y)
        coords_arr = np.asarray(validate_coords(coords), dtype=float)
        if attributes is None:
            attrs_arr = np.zeros((coords_arr.shape[0], 0), dtype=float)
            attr_names: Tuple[str, ...] = ()
        else:
            attrs_arr = self._numeric_2d(attributes, "attributes")
            attr_names = self._input_names(attributes, "attr", attrs_arr.shape[1])

        n = X_raw.shape[0]
        if y_arr.shape[0] != n or coords_arr.shape[0] != n or attrs_arr.shape[0] != n:
            raise ValueError("X, y, coords and attributes must contain the same rows.")
        X_design = add_intercept(X_raw) if self.fit_intercept else X_raw.copy()
        if n <= X_design.shape[1] + 1:
            raise ValueError(
                "LG-GWR needs more observations than local design parameters plus one."
            )
        if self.scale_constraint == "orthogonal" and self.latent_dim > (
            coords_arr.shape[1] + attrs_arr.shape[1]
        ):
            raise ValueError(
                "orthogonal scale_constraint requires latent_dim <= geometry "
                "input dimension."
            )

        coord_names = self._input_names(coords, "coord", coords_arr.shape[1])
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.feature_names_ = feature_names
        self.geometry_feature_names_ = coord_names + attr_names
        self.n_features_in_ = X_raw.shape[1]
        self.X_train_ = X_raw.copy()
        self.X_design_ = X_design.copy()
        self.y_train_ = y_arr.copy()
        self.coords_train_ = coords_arr.copy()
        self.attrs_train_ = attrs_arr.copy()
        self.coords_geometry_, self.attrs_geometry_ = self._fit_geometry_scaler(
            coords_arr, attrs_arr
        )
        self.u_train_ = np.hstack([self.coords_geometry_, self.attrs_geometry_])
        return X_design, y_arr, self.coords_geometry_, self.attrs_geometry_

    def _prepare_prediction_inputs(
        self,
        X: ArrayLike,
        coords: ArrayLike,
        attributes: Optional[ArrayLike],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()
        X_raw = self._coerce_X_predict(X)
        X_design = add_intercept(X_raw) if self.fit_intercept else X_raw.copy()
        coords_arr = np.asarray(validate_coords(coords), dtype=float)
        if attributes is None:
            attrs_arr = np.zeros((coords_arr.shape[0], 0), dtype=float)
        else:
            attrs_arr = self._numeric_2d(attributes, "attributes")
        if (
            X_raw.shape[0] != coords_arr.shape[0]
            or attrs_arr.shape[0] != X_raw.shape[0]
        ):
            raise ValueError("X, coords and attributes must contain the same rows.")
        coords_geometry, attrs_geometry = self._transform_geometry(
            coords_arr, attrs_arr
        )
        return X_design, coords_arr, coords_geometry, attrs_geometry

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("LGGWR is not fitted. Call fit() first.")

    # ------------------------------------------------------------------
    # Kernels and latent maps
    # ------------------------------------------------------------------
    def _kernel_weights(self, d: np.ndarray, h: float) -> np.ndarray:
        """Return kernel weights for non-negative distances."""
        if np.isinf(h):
            return np.ones_like(d, dtype=float)
        v = d / h
        if self.kernel == "gaussian":
            return np.exp(-0.5 * v**2)
        if self.kernel == "bisquare":
            return np.where(v < 1.0, (1.0 - v**2) ** 2, 0.0)
        return np.exp(-v)

    def _kernel_deriv_over_d(self, d: np.ndarray, h: float) -> np.ndarray:
        """Return ``(dK/dd) / d`` in a numerically stable closed form."""
        if np.isinf(h):
            return np.zeros_like(d, dtype=float)
        eps = 1e-12
        v = d / h
        if self.kernel == "gaussian":
            return -(1.0 / h**2) * np.exp(-0.5 * v**2)
        if self.kernel == "bisquare":
            return np.where(v < 1.0, -(4.0 / h**2) * (1.0 - v**2), 0.0)
        ratio = np.zeros_like(d)
        mask = d > eps
        ratio[mask] = -(1.0 / h) * np.exp(-v[mask]) / d[mask]
        return ratio

    def _initialize_A(
        self,
        input_dim: int,
        rng: np.random.Generator,
        coord_dim: Optional[int] = None,
        u: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Initialise the joint latent map.

        ``coord_dim`` and ``u`` are optional for backward compatibility with the
        original private numerical-gradient tests.
        """
        mode = self.initialization if mode is None else mode
        scale = np.sqrt(2.0 / (input_dim + self.latent_dim))
        if mode == "coordinate" and coord_dim is not None:
            A = np.zeros((self.latent_dim, input_dim), dtype=float)
            diagonal = min(self.latent_dim, coord_dim)
            A[np.arange(diagonal), np.arange(diagonal)] = 1.0
            if self.latent_dim > diagonal:
                A[diagonal:, :] = (
                    rng.standard_normal((self.latent_dim - diagonal, input_dim)) * scale
                )
        elif mode == "pca" and u is not None:
            _, _, vt = np.linalg.svd(u - np.mean(u, axis=0), full_matrices=False)
            rows = min(self.latent_dim, vt.shape[0])
            A = np.zeros((self.latent_dim, input_dim), dtype=float)
            A[:rows, :] = vt[:rows, :]
            if rows < self.latent_dim:
                A[rows:, :] = (
                    rng.standard_normal((self.latent_dim - rows, input_dim)) * scale
                )
        else:
            A = rng.standard_normal((self.latent_dim, input_dim)) * scale
        return self._project_matrix(A, np.linalg.norm(A, "fro"))

    def _initialize_B(
        self,
        q: int,
        rng: np.random.Generator,
        attrs: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Initialise the separable attribute map."""
        if q == 0:
            return np.zeros((self.latent_dim, 0), dtype=float)
        mode = self.initialization if mode is None else mode
        scale = np.sqrt(2.0 / (q + self.latent_dim))
        if mode == "pca" and attrs is not None:
            _, _, vt = np.linalg.svd(
                attrs - np.mean(attrs, axis=0), full_matrices=False
            )
            rows = min(self.latent_dim, vt.shape[0])
            B = np.zeros((self.latent_dim, q), dtype=float)
            B[:rows, :] = vt[:rows, :]
            if rows < self.latent_dim:
                B[rows:, :] = rng.standard_normal((self.latent_dim - rows, q)) * scale
        else:
            B = rng.standard_normal((self.latent_dim, q)) * scale
        return self._project_matrix(B, np.linalg.norm(B, "fro"))

    def _project_matrix(self, matrix: np.ndarray, target_norm: float) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        if self.scale_constraint == "orthogonal":
            u, _, vt = np.linalg.svd(matrix, full_matrices=False)
            return u @ vt
        if self.scale_constraint == "frobenius":
            current = np.linalg.norm(matrix, "fro")
            if current > 1e-12:
                return matrix * (target_norm / current)
        return matrix

    def _compute_latent_coords(self, u: np.ndarray) -> np.ndarray:
        """Map joint geometry input to latent coordinates."""
        if self.A_ is None:
            raise RuntimeError("Joint transformation matrix A is unavailable.")
        return u @ self.A_.T

    @staticmethod
    def _auto_distance_bandwidth(
        distance_matrix: np.ndarray, n_parameters: int
    ) -> float:
        n = distance_matrix.shape[0]
        sorted_distances = np.sort(distance_matrix, axis=1)
        neighbour = min(max(n_parameters + 2, int(np.sqrt(n))), n - 1)
        value = float(np.median(sorted_distances[:, neighbour]))
        if not np.isfinite(value) or value <= 1e-12:
            positive = distance_matrix[distance_matrix > 1e-12]
            value = float(np.median(positive)) if positive.size else 1.0
        return max(value, 1e-6)

    def _resolve_bandwidth(self, z_init: np.ndarray, n_features: int) -> float:
        """Resolve a fixed latent-space distance bandwidth."""
        distances = cdist(z_init, z_init, metric="euclidean")
        if self.bandwidth is not None and not isinstance(self.bandwidth, tuple):
            if not self.adaptive:
                return float(self.bandwidth)
            k = min(int(self.bandwidth), z_init.shape[0] - 1)
            warnings.warn(
                "LG-GWR analytical training uses a fixed distance bandwidth; the "
                "adaptive neighbour count is converted to the median k-th distance.",
                UserWarning,
                stacklevel=2,
            )
            return max(float(np.median(np.sort(distances, axis=1)[:, k])), 1e-6)
        return self._auto_distance_bandwidth(distances, n_features)

    # ------------------------------------------------------------------
    # Local weighted least squares and analytical gradient
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_wls(M: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve a local normal equation with deterministic fallbacks."""
        p = M.shape[0]
        try:
            beta = np.linalg.solve(M, b)
            if np.all(np.isfinite(beta)):
                return beta
        except np.linalg.LinAlgError:
            pass
        ridge = 1e-6 * (np.trace(M) / max(p, 1) + 1e-12) + 1e-12
        try:
            beta = np.linalg.solve(M + ridge * np.eye(p), b)
            if np.all(np.isfinite(beta)):
                return beta
        except np.linalg.LinAlgError:
            pass
        return np.linalg.pinv(M) @ b

    @classmethod
    def _hat_solution(
        cls, X: np.ndarray, y: np.ndarray, weights: np.ndarray, x_query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xw = X * weights[:, None]
        M = Xw.T @ X
        C = cls._solve_wls(M, Xw.T)
        beta = C @ y
        hat_row = x_query @ C
        return beta, hat_row

    def _forward_loo(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray, h: float
    ) -> Dict[str, np.ndarray]:
        """Run the joint leave-one-out local regressions."""
        n, p = X.shape
        distances = cdist(z, z, metric="euclidean")
        weights = self._kernel_weights(distances, h)
        np.fill_diagonal(weights, 0.0)
        beta = np.zeros((n, p))
        g = np.zeros((n, p))
        yhat = np.zeros(n)
        for i in range(n):
            Xw = X * weights[i, :, None]
            M = Xw.T @ X
            beta[i] = self._solve_wls(M, Xw.T @ y)
            g[i] = self._solve_wls(M, X[i])
            yhat[i] = X[i] @ beta[i]
        return {
            "d": distances,
            "W": weights,
            "beta": beta,
            "g": g,
            "yhat": yhat,
        }

    def _compute_loss(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """Return LOO mean squared error plus optional unconstrained L2 penalty."""
        if self.A_ is None:
            raise RuntimeError("A is unavailable for joint loss computation.")
        return float(np.mean((y - yhat) ** 2) + self.lambda_reg * np.sum(self.A_**2))

    def _compute_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        z: np.ndarray,
        h: float,
        cache: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return the analytical gradient of the joint LOO objective."""
        if self.A_ is None:
            raise RuntimeError("A is unavailable for joint gradient computation.")
        n = X.shape[0]
        residual = y - cache["yhat"]
        ratio = self._kernel_deriv_over_d(cache["d"], h)
        np.fill_diagonal(ratio, 0.0)
        gradient = np.zeros_like(self.A_)
        for i in range(n):
            sensitivity = X @ cache["g"][i]
            local_error = y - X @ cache["beta"][i]
            coefficient = residual[i] * sensitivity * local_error * ratio[i]
            z_diff = z[i][None, :] - z
            u_diff = u[i][None, :] - u
            gradient += (z_diff * coefficient[:, None]).T @ u_diff
        return -(2.0 / n) * gradient + 2.0 * self.lambda_reg * self.A_

    def _forward_loo_sep(
        self,
        X: np.ndarray,
        y: np.ndarray,
        geographic_weights: np.ndarray,
        zeta: np.ndarray,
        h_a: float,
    ) -> Dict[str, np.ndarray]:
        """Run separable leave-one-out local regressions."""
        n, p = X.shape
        attr_distances = (
            cdist(zeta, zeta, metric="euclidean") if zeta.shape[1] else np.zeros((n, n))
        )
        attribute_weights = (
            self._kernel_weights(attr_distances, h_a)
            if zeta.shape[1]
            else np.ones((n, n))
        )
        weights = geographic_weights * attribute_weights
        np.fill_diagonal(weights, 0.0)
        beta = np.zeros((n, p))
        g = np.zeros((n, p))
        yhat = np.zeros(n)
        for i in range(n):
            Xw = X * weights[i, :, None]
            M = Xw.T @ X
            beta[i] = self._solve_wls(M, Xw.T @ y)
            g[i] = self._solve_wls(M, X[i])
            yhat[i] = X[i] @ beta[i]
        return {
            "da": attr_distances,
            "Kg": geographic_weights,
            "beta": beta,
            "g": g,
            "yhat": yhat,
        }

    def _compute_gradient_sep(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attributes: np.ndarray,
        zeta: np.ndarray,
        h_a: float,
        cache: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return the analytical gradient of the separable LOO objective."""
        if self.B_ is None:
            raise RuntimeError("B is unavailable for separable gradient computation.")
        n = X.shape[0]
        residual = y - cache["yhat"]
        ratio = self._kernel_deriv_over_d(cache["da"], h_a)
        np.fill_diagonal(ratio, 0.0)
        gradient = np.zeros_like(self.B_)
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n):
                sensitivity = X @ cache["g"][i]
                local_error = y - X @ cache["beta"][i]
                coefficient = (
                    residual[i] * sensitivity * local_error * cache["Kg"][i] * ratio[i]
                )
                z_diff = zeta[i][None, :] - zeta
                a_diff = attributes[i][None, :] - attributes
                gradient += (z_diff * coefficient[:, None]).T @ a_diff
        return -(2.0 / n) * gradient + 2.0 * self.lambda_reg * self.B_

    def _local_fit_with_hat(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray, h: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit joint local models at training locations and return the hat matrix."""
        n, p = X.shape
        weights = self._kernel_weights(cdist(z, z), h)
        betas = np.zeros((n, p))
        hat = np.zeros((n, n))
        for i in range(n):
            betas[i], hat[i] = self._hat_solution(X, y, weights[i], X[i])
        return betas, hat

    def _local_fit_with_hat_sep(
        self,
        X: np.ndarray,
        y: np.ndarray,
        geographic_distances: np.ndarray,
        zeta: np.ndarray,
        h_g: float,
        h_a: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit separable training models and return their hat matrix."""
        geographic_weights = self._kernel_weights(geographic_distances, h_g)
        attribute_weights = (
            self._kernel_weights(cdist(zeta, zeta), h_a)
            if zeta.shape[1]
            else np.ones_like(geographic_weights)
        )
        weights = geographic_weights * attribute_weights
        n, p = X.shape
        betas = np.zeros((n, p))
        hat = np.zeros((n, n))
        for i in range(n):
            betas[i], hat[i] = self._hat_solution(X, y, weights[i], X[i])
        return betas, hat

    def _local_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        z_train: np.ndarray,
        z_query: np.ndarray,
        bandwidth: float,
        X_query: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Recalibrate joint local coefficients at query latent locations."""
        n_query = z_query.shape[0]
        p = X_train.shape[1]
        weights = self._kernel_weights(cdist(z_query, z_train), bandwidth)
        global_beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        betas = np.zeros((n_query, p))
        for index in range(n_query):
            if np.sum(weights[index] > 1e-8) < p:
                betas[index] = global_beta
                continue
            query_design = X_query[index] if X_query is not None else X_train[0]
            beta, _ = self._hat_solution(X_train, y_train, weights[index], query_design)
            betas[index] = beta if np.all(np.isfinite(beta)) else global_beta
        return betas

    def _local_fit_sep(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        coords_train: np.ndarray,
        zeta_train: np.ndarray,
        coords_query: np.ndarray,
        zeta_query: np.ndarray,
        h_g: float,
        h_a: float,
        X_query: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Recalibrate separable local coefficients at query locations."""
        geographic_weights = self._kernel_weights(
            cdist(coords_query, coords_train), h_g
        )
        attribute_weights = (
            self._kernel_weights(cdist(zeta_query, zeta_train), h_a)
            if zeta_train.shape[1]
            else np.ones_like(geographic_weights)
        )
        weights = geographic_weights * attribute_weights
        n_query = coords_query.shape[0]
        p = X_train.shape[1]
        global_beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        betas = np.zeros((n_query, p))
        for index in range(n_query):
            if np.sum(weights[index] > 1e-8) < p:
                betas[index] = global_beta
                continue
            query_design = X_query[index] if X_query is not None else X_train[0]
            beta, _ = self._hat_solution(X_train, y_train, weights[index], query_design)
            betas[index] = beta if np.all(np.isfinite(beta)) else global_beta
        return betas

    # ------------------------------------------------------------------
    # Optimisation and bandwidth selection
    # ------------------------------------------------------------------
    def _optimise_joint(
        self,
        X: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        initial: np.ndarray,
        bandwidth: float,
    ) -> _OptimisationResult:
        self.A_ = initial.copy()
        target_norm = np.linalg.norm(initial, "fro")
        first_loss: Optional[float] = None
        best_loss = np.inf
        best_matrix = initial.copy()
        history: list[float] = []
        m = np.zeros_like(initial)
        v = np.zeros_like(initial)
        previous = np.inf
        stalled = 0
        converged = False
        stop_reason = "max_iter"
        for iteration in range(self.max_iter):
            z = u @ self.A_.T
            cache = self._forward_loo(X, y, z, bandwidth)
            loss = self._compute_loss(y, cache["yhat"])
            if first_loss is None:
                first_loss = loss
            if not np.isfinite(loss):
                stop_reason = "nonfinite_loss"
                break
            history.append(float(loss))
            if loss < best_loss - self.tol:
                best_loss = float(loss)
                best_matrix = self.A_.copy()
                stalled = 0
            else:
                stalled += 1
            if stalled >= self.patience:
                stop_reason = "patience"
                converged = True
                break
            if iteration > 0 and abs(previous - loss) < self.tol:
                stop_reason = "tolerance"
                converged = True
                break
            previous = loss
            gradient = self._compute_gradient(X, y, u, z, bandwidth, cache)
            if not np.all(np.isfinite(gradient)):
                stop_reason = "nonfinite_gradient"
                break
            norm = np.linalg.norm(gradient)
            if norm > self.grad_clip:
                gradient *= self.grad_clip / norm
            step = iteration + 1
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient**2
            m_hat = m / (1.0 - 0.9**step)
            v_hat = v / (1.0 - 0.999**step)
            self.A_ -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.A_ = self._project_matrix(self.A_, target_norm)
        self.A_ = best_matrix
        final_cache = self._forward_loo(X, y, u @ self.A_.T, bandwidth)
        final_loss = self._compute_loss(y, final_cache["yhat"])
        if not history:
            history = [float(final_loss)]
            best_loss = float(final_loss)
        return _OptimisationResult(
            matrix=self.A_.copy(),
            loss_history=tuple(history),
            best_loss=float(best_loss),
            final_loss=float(final_loss),
            n_iter=len(history),
            converged=bool(converged),
            stop_reason=stop_reason,
        )

    def _optimise_separable(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attributes: np.ndarray,
        initial: np.ndarray,
        geographic_weights: np.ndarray,
        h_a: float,
    ) -> _OptimisationResult:
        self.B_ = initial.copy()
        if initial.size == 0:
            cache = self._forward_loo_sep(
                X, y, geographic_weights, np.zeros((X.shape[0], 0)), h_a
            )
            loss = float(np.mean((y - cache["yhat"]) ** 2))
            return _OptimisationResult(
                matrix=initial.copy(),
                loss_history=(loss,),
                best_loss=loss,
                final_loss=loss,
                n_iter=0,
                converged=True,
                stop_reason="no_attributes",
            )
        target_norm = np.linalg.norm(initial, "fro")
        best_loss = np.inf
        best_matrix = initial.copy()
        history: list[float] = []
        m = np.zeros_like(initial)
        v = np.zeros_like(initial)
        previous = np.inf
        stalled = 0
        converged = False
        stop_reason = "max_iter"
        for iteration in range(self.max_iter):
            zeta = attributes @ self.B_.T
            cache = self._forward_loo_sep(X, y, geographic_weights, zeta, h_a)
            loss = float(
                np.mean((y - cache["yhat"]) ** 2) + self.lambda_reg * np.sum(self.B_**2)
            )
            if not np.isfinite(loss):
                stop_reason = "nonfinite_loss"
                break
            history.append(loss)
            if loss < best_loss - self.tol:
                best_loss = loss
                best_matrix = self.B_.copy()
                stalled = 0
            else:
                stalled += 1
            if stalled >= self.patience:
                stop_reason = "patience"
                converged = True
                break
            if iteration > 0 and abs(previous - loss) < self.tol:
                stop_reason = "tolerance"
                converged = True
                break
            previous = loss
            gradient = self._compute_gradient_sep(X, y, attributes, zeta, h_a, cache)
            if not np.all(np.isfinite(gradient)):
                stop_reason = "nonfinite_gradient"
                break
            norm = np.linalg.norm(gradient)
            if norm > self.grad_clip:
                gradient *= self.grad_clip / norm
            step = iteration + 1
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient**2
            m_hat = m / (1.0 - 0.9**step)
            v_hat = v / (1.0 - 0.999**step)
            self.B_ -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.B_ = self._project_matrix(self.B_, target_norm)
        self.B_ = best_matrix
        final_cache = self._forward_loo_sep(
            X, y, geographic_weights, attributes @ self.B_.T, h_a
        )
        final_loss = float(
            np.mean((y - final_cache["yhat"]) ** 2)
            + self.lambda_reg * np.sum(self.B_**2)
        )
        if not history:
            history = [final_loss]
            best_loss = final_loss
        return _OptimisationResult(
            matrix=self.B_.copy(),
            loss_history=tuple(history),
            best_loss=float(best_loss),
            final_loss=float(final_loss),
            n_iter=len(history),
            converged=bool(converged),
            stop_reason=stop_reason,
        )

    def _select_bandwidth_aicc(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray, n_grid: int = 16
    ) -> float:
        distances = cdist(z, z)
        n, p = X.shape
        sorted_distances = np.sort(distances, axis=1)
        lower = max(float(np.median(sorted_distances[:, min(p + 2, n - 1)])), 1e-6)
        upper = max(float(np.max(distances)), lower * 4.0)
        candidates = list(np.geomspace(lower, upper, n_grid))
        if isinstance(self.bandwidth_, Real):
            candidates.append(float(self.bandwidth_))
        best_bandwidth = candidates[0]
        best_aicc = np.inf
        for candidate in candidates:
            betas, hat = self._local_fit_with_hat(X, y, z, float(candidate))
            fitted = np.einsum("ij,ij->i", X, betas)
            aicc = compute_diagnostics(
                y, fitted, hat_matrix=hat, compute_gwr_stats=True
            )["aicc"]
            if np.isfinite(aicc) and aicc < best_aicc:
                best_aicc = float(aicc)
                best_bandwidth = float(candidate)
        return best_bandwidth

    def _select_bandwidths_aicc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        geographic_distances: np.ndarray,
        zeta: np.ndarray,
        current: Optional[Tuple[float, float]] = None,
        n_grid: int = 7,
    ) -> Tuple[float, float]:
        n, p = X.shape

        def bounds(distance_matrix: np.ndarray) -> Tuple[float, float]:
            sorted_distances = np.sort(distance_matrix, axis=1)
            lower = max(float(np.median(sorted_distances[:, min(p + 2, n - 1)])), 1e-6)
            upper = max(float(np.max(distance_matrix)), lower * 4.0)
            return lower, upper

        g_lower, g_upper = bounds(geographic_distances)
        g_grid = list(np.geomspace(g_lower, g_upper, n_grid))
        if zeta.shape[1]:
            a_lower, a_upper = bounds(cdist(zeta, zeta))
            a_grid: list[float] = list(np.geomspace(a_lower, a_upper, n_grid))
            a_grid.append(np.inf)
        else:
            a_grid = [np.inf]
        if current is not None:
            g_grid.append(float(current[0]))
            a_grid.append(float(current[1]))

        def score(h_g: float, h_a: float) -> float:
            betas, hat = self._local_fit_with_hat_sep(
                X, y, geographic_distances, zeta, h_g, h_a
            )
            fitted = np.einsum("ij,ij->i", X, betas)
            value = compute_diagnostics(
                y, fitted, hat_matrix=hat, compute_gwr_stats=True
            )["aicc"]
            return float(value) if np.isfinite(value) else np.inf

        h_g = g_grid[len(g_grid) // 2] if current is None else float(current[0])
        h_a = np.inf if current is None else float(current[1])
        for _ in range(2):
            h_g = min(g_grid, key=lambda candidate: score(float(candidate), h_a))
            h_a = min(a_grid, key=lambda candidate: score(float(h_g), float(candidate)))
        return float(h_g), float(h_a)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike,
        y: VectorLike,
        coords: ArrayLike,
        attributes: Optional[ArrayLike] = None,
    ) -> "LGGWR":
        """Fit LG-GWR and return ``self``."""
        self._reset_fit_state()
        try:
            X_design, y_arr, coords_geometry, attrs_geometry = self._prepare_fit_inputs(
                X, y, coords, attributes
            )
            if self.geometry == "separable":
                self._fit_separable(X_design, y_arr, coords_geometry, attrs_geometry)
            else:
                self._fit_joint(X_design, y_arr, coords_geometry, attrs_geometry)
            self._finalise_public_parameters()
            self._is_fitted = True
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def _fit_joint(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        attrs: np.ndarray,
    ) -> None:
        u = np.hstack([coords, attrs])
        input_dim = u.shape[1]
        coord_dim = coords.shape[1]
        restart_records: list[Dict[str, Any]] = []
        base_seed = 0 if self.random_state is None else int(self.random_state)

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(base_seed + restart)
            mode = self.initialization if restart == 0 else "random"
            initial = self._initialize_A(input_dim, rng, coord_dim, u, mode)
            self.A_ = initial.copy()
            bandwidth = self._resolve_bandwidth(u @ self.A_.T, X.shape[1])
            bandwidth_history: list[float] = [float(bandwidth)]
            stage_history: list[float] = []
            stage_result: Optional[_OptimisationResult] = None
            n_stages = self.bandwidth_updates + 1 if self.select_bandwidth else 1
            for stage in range(n_stages):
                stage_result = self._optimise_joint(X, y, u, self.A_, bandwidth)
                self.A_ = stage_result.matrix.copy()
                stage_history.extend(stage_result.loss_history)
                if self.select_bandwidth:
                    self.bandwidth_ = bandwidth
                    selected = self._select_bandwidth_aicc(X, y, u @ self.A_.T)
                    bandwidth_history.append(float(selected))
                    bandwidth = float(selected)
                if stage == n_stages - 1:
                    break
            assert stage_result is not None
            z = u @ self.A_.T
            betas, hat = self._local_fit_with_hat(X, y, z, bandwidth)
            fitted = np.einsum("ij,ij->i", X, betas)
            diagnostics = compute_diagnostics(
                y, fitted, hat_matrix=hat, compute_gwr_stats=True
            )
            final_cache = self._forward_loo(X, y, z, bandwidth)
            final_loo = self._compute_loss(y, final_cache["yhat"])
            restart_records.append(
                {
                    "matrix": self.A_.copy(),
                    "bandwidth": bandwidth,
                    "bandwidth_history": bandwidth_history,
                    "loss_history": stage_history,
                    "best_loss": min(stage_history) if stage_history else final_loo,
                    "final_loo": final_loo,
                    "n_iter": len(stage_history),
                    "converged": stage_result.converged,
                    "stop_reason": stage_result.stop_reason,
                    "betas": betas,
                    "hat": hat,
                    "fitted": fitted,
                    "diagnostics": diagnostics,
                }
            )

        best = min(
            restart_records,
            key=lambda record: (record["diagnostics"]["aicc"], record["final_loo"]),
        )
        self.restart_scores_ = [
            {
                "restart": float(index),
                "aicc": float(record["diagnostics"]["aicc"]),
                "final_loo_loss": float(record["final_loo"]),
            }
            for index, record in enumerate(restart_records)
        ]
        self.A_ = best["matrix"]
        self.B_ = None
        self.bandwidth_ = float(best["bandwidth"])
        self.bandwidth_history_ = list(best["bandwidth_history"])
        self.loss_history_ = list(best["loss_history"])
        self.best_loss_ = float(best["best_loss"])
        self.final_loo_loss_ = float(best["final_loo"])
        self.n_iter_ = int(best["n_iter"])
        self.converged_ = bool(best["converged"])
        self.stop_reason_ = str(best["stop_reason"])
        self.latent_coords_ = u @ self.A_.T
        self.coefficients_ = best["betas"]
        self.hat_matrix_ = best["hat"]
        self.fitted_values_ = best["fitted"]
        self.residuals_ = y - self.fitted_values_
        self.diagnostics_ = best["diagnostics"]
        self._set_metric_outputs(self.A_)

    def _fit_separable(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        attrs: np.ndarray,
    ) -> None:
        geographic_distances = cdist(coords, coords)
        if isinstance(self.bandwidth, tuple):
            initial_bandwidth = (float(self.bandwidth[0]), float(self.bandwidth[1]))
        else:
            h_g = (
                float(self.bandwidth)
                if self.bandwidth is not None
                else self._auto_distance_bandwidth(geographic_distances, X.shape[1])
            )
            h_a = np.inf
            initial_bandwidth = (h_g, h_a)

        restart_records: list[Dict[str, Any]] = []
        base_seed = 0 if self.random_state is None else int(self.random_state)
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(base_seed + restart)
            mode = self.initialization if restart == 0 else "random"
            self.B_ = self._initialize_B(attrs.shape[1], rng, attrs, mode)
            h_g, h_a = initial_bandwidth
            if attrs.shape[1] and np.isinf(h_a):
                initial_attr_distances = cdist(attrs @ self.B_.T, attrs @ self.B_.T)
                h_a = self._auto_distance_bandwidth(initial_attr_distances, X.shape[1])
            bandwidth_history: list[Tuple[float, float]] = [(float(h_g), float(h_a))]
            stage_history: list[float] = []
            stage_result: Optional[_OptimisationResult] = None
            n_stages = self.bandwidth_updates + 1 if self.select_bandwidth else 1
            for stage in range(n_stages):
                geographic_weights = self._kernel_weights(geographic_distances, h_g)
                stage_result = self._optimise_separable(
                    X, y, attrs, self.B_, geographic_weights, h_a
                )
                self.B_ = stage_result.matrix.copy()
                stage_history.extend(stage_result.loss_history)
                zeta = attrs @ self.B_.T if attrs.shape[1] else np.zeros((len(y), 0))
                if self.select_bandwidth:
                    h_g, h_a = self._select_bandwidths_aicc(
                        X,
                        y,
                        geographic_distances,
                        zeta,
                        current=(h_g, h_a),
                    )
                    bandwidth_history.append((float(h_g), float(h_a)))
                if stage == n_stages - 1:
                    break
            assert stage_result is not None
            zeta = attrs @ self.B_.T if attrs.shape[1] else np.zeros((len(y), 0))
            betas, hat = self._local_fit_with_hat_sep(
                X, y, geographic_distances, zeta, h_g, h_a
            )
            fitted = np.einsum("ij,ij->i", X, betas)
            diagnostics = compute_diagnostics(
                y, fitted, hat_matrix=hat, compute_gwr_stats=True
            )
            final_cache = self._forward_loo_sep(
                X, y, self._kernel_weights(geographic_distances, h_g), zeta, h_a
            )
            final_loo = float(
                np.mean((y - final_cache["yhat"]) ** 2)
                + self.lambda_reg * np.sum(self.B_**2)
            )
            restart_records.append(
                {
                    "matrix": self.B_.copy(),
                    "bandwidth": (float(h_g), float(h_a)),
                    "bandwidth_history": bandwidth_history,
                    "loss_history": stage_history,
                    "best_loss": min(stage_history) if stage_history else final_loo,
                    "final_loo": final_loo,
                    "n_iter": len(stage_history),
                    "converged": stage_result.converged,
                    "stop_reason": stage_result.stop_reason,
                    "betas": betas,
                    "hat": hat,
                    "fitted": fitted,
                    "diagnostics": diagnostics,
                    "zeta": zeta,
                }
            )

        best = min(
            restart_records,
            key=lambda record: (record["diagnostics"]["aicc"], record["final_loo"]),
        )
        self.restart_scores_ = [
            {
                "restart": float(index),
                "aicc": float(record["diagnostics"]["aicc"]),
                "final_loo_loss": float(record["final_loo"]),
            }
            for index, record in enumerate(restart_records)
        ]
        self.A_ = None
        self.B_ = best["matrix"]
        self.bandwidth_ = best["bandwidth"]
        self.bandwidth_history_ = list(best["bandwidth_history"])
        self.loss_history_ = list(best["loss_history"])
        self.best_loss_ = float(best["best_loss"])
        self.final_loo_loss_ = float(best["final_loo"])
        self.n_iter_ = int(best["n_iter"])
        self.converged_ = bool(best["converged"])
        self.stop_reason_ = str(best["stop_reason"])
        self.latent_coords_ = best["zeta"]
        self.coefficients_ = best["betas"]
        self.hat_matrix_ = best["hat"]
        self.fitted_values_ = best["fitted"]
        self.residuals_ = y - self.fitted_values_
        self.diagnostics_ = best["diagnostics"]
        self._set_metric_outputs(self.B_)

    def _set_metric_outputs(self, matrix: np.ndarray) -> None:
        self.metric_matrix_ = matrix.T @ matrix
        diagonal = np.clip(np.diag(self.metric_matrix_), 0.0, np.inf)
        total = float(np.sum(diagonal))
        self.metric_contributions_ = (
            diagonal / total if total > 0.0 else np.zeros_like(diagonal)
        )

    def _finalise_public_parameters(self) -> None:
        if self.coefficients_ is None:
            raise RuntimeError("Local parameters are unavailable.")
        if self.fit_intercept:
            self.intercept_ = self.coefficients_[:, 0].copy()
            self.coef_ = self.coefficients_[:, 1:].copy()
        else:
            self.intercept_ = np.zeros(self.coefficients_.shape[0], dtype=float)
            self.coef_ = self.coefficients_.copy()

    # ------------------------------------------------------------------
    # Prediction and reporting
    # ------------------------------------------------------------------
    def predict_result(
        self,
        X: ArrayLike,
        coords: ArrayLike,
        attributes: Optional[ArrayLike] = None,
    ) -> LGGWRPredictionResult:
        """Recalibrate local parameters at new locations."""
        X_design, coords_raw, coords_geometry, attrs_geometry = (
            self._prepare_prediction_inputs(X, coords, attributes)
        )
        if self.X_design_ is None or self.y_train_ is None:
            raise RuntimeError("Training state is incomplete.")

        if self.geometry == "separable":
            if self.B_ is None or not isinstance(self.bandwidth_, tuple):
                raise RuntimeError("Separable training state is incomplete.")
            zeta_train = (
                self.attrs_geometry_ @ self.B_.T
                if self.attrs_geometry_ is not None and self.attrs_geometry_.shape[1]
                else np.zeros((self.X_design_.shape[0], 0))
            )
            zeta_query = (
                attrs_geometry @ self.B_.T
                if attrs_geometry.shape[1]
                else np.zeros((X_design.shape[0], 0))
            )
            betas = self._local_fit_sep(
                self.X_design_,
                self.y_train_,
                self.coords_geometry_,
                zeta_train,
                coords_geometry,
                zeta_query,
                float(self.bandwidth_[0]),
                float(self.bandwidth_[1]),
                X_design,
            )
            latent = zeta_query
        else:
            if self.A_ is None or not isinstance(self.bandwidth_, Real):
                raise RuntimeError("Joint training state is incomplete.")
            u_query = np.hstack([coords_geometry, attrs_geometry])
            latent = u_query @ self.A_.T
            betas = self._local_fit(
                self.X_design_,
                self.y_train_,
                self.latent_coords_,
                latent,
                float(self.bandwidth_),
                X_design,
            )
        predictions = np.einsum("ij,ij->i", X_design, betas)
        if self.fit_intercept:
            intercepts = betas[:, 0]
            coefficients = betas[:, 1:]
        else:
            intercepts = np.zeros(X_design.shape[0], dtype=float)
            coefficients = betas
        return LGGWRPredictionResult(
            predictions=predictions,
            coefficients=coefficients,
            intercepts=intercepts,
            coords=coords_raw.copy(),
            latent_coords=latent.copy(),
            feature_names=self.feature_names_,
        )

    def predict(
        self,
        X: ArrayLike,
        coords: ArrayLike,
        attributes: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return LG-GWR predictions at new locations."""
        return self.predict_result(X, coords, attributes).predictions

    def results_frame(self) -> pd.DataFrame:
        """Return training-location parameters, fitted values and latent coordinates."""
        self._require_fitted()
        if (
            self.coords_train_ is None
            or self.fitted_values_ is None
            or self.residuals_ is None
            or self.coef_ is None
            or self.intercept_ is None
            or self.latent_coords_ is None
        ):
            raise RuntimeError("Training results are incomplete.")
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords_train_[:, 0],
            "coord_1": self.coords_train_[:, 1],
            "fitted": self.fitted_values_,
            "residual": self.residuals_,
            "intercept": self.intercept_,
        }
        for index in range(self.latent_coords_.shape[1]):
            data[f"latent_{index}"] = self.latent_coords_[:, index]
        for index, name in enumerate(self.feature_names_):
            data[f"coef_{name}"] = self.coef_[:, index]
        return pd.DataFrame(data)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`results_frame`."""
        return self.results_frame()

    def metric_frame(self) -> pd.DataFrame:
        """Return rotation-invariant learned metric contributions."""
        self._require_fitted()
        if self.metric_matrix_ is None or self.metric_contributions_ is None:
            raise RuntimeError("Metric outputs are unavailable.")
        if self.geometry == "joint":
            names = self.geometry_feature_names_
        else:
            coord_count = 0 if self.coord_center_ is None else self.coord_center_.size
            names = self.geometry_feature_names_[coord_count:]
        return pd.DataFrame(
            {
                "geometry_feature": list(names),
                "metric_diagonal": np.diag(self.metric_matrix_),
                "metric_contribution": self.metric_contributions_,
            }
        )

    def get_latent_coordinates(
        self,
        coords: Optional[ArrayLike] = None,
        attributes: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return training or transformed latent coordinates."""
        self._require_fitted()
        if coords is None:
            return self.latent_coords_.copy()
        n = np.asarray(coords).shape[0]
        dummy = np.zeros((n, self.n_features_in_), dtype=float)
        _, _, coords_geometry, attrs_geometry = self._prepare_prediction_inputs(
            dummy, coords, attributes
        )
        if self.geometry == "separable":
            if self.B_ is None:
                raise RuntimeError("B is unavailable.")
            return (
                attrs_geometry @ self.B_.T
                if attrs_geometry.shape[1]
                else np.zeros((n, 0))
            )
        if self.A_ is None:
            raise RuntimeError("A is unavailable.")
        return np.hstack([coords_geometry, attrs_geometry]) @ self.A_.T

    def summary(self) -> str:
        """Return a plain-text fitted-model summary."""
        self._require_fitted()
        if self.diagnostics_ is None or self.metric_contributions_ is None:
            raise RuntimeError("Model diagnostics are incomplete.")
        matrix = self.A_ if self.geometry == "joint" else self.B_
        return format_summary(
            "LG-GWR Summary",
            {
                "model": "LG-GWR",
                "geometry": self.geometry,
                "n_samples": int(self.y_train_.size),
                "n_features": int(self.n_features_in_),
                "latent_dim": int(self.latent_dim),
                "bandwidth": self.bandwidth_,
                "bandwidth_history": tuple(self.bandwidth_history_),
                "kernel": self.kernel,
                "fit_intercept": self.fit_intercept,
                "standardize_geometry": self.standardize_geometry,
                "initialization": self.initialization,
                "n_restarts": self.n_restarts,
                "scale_constraint": self.scale_constraint,
                "n_iterations": self.n_iter_,
                "converged": self.converged_,
                "stop_reason": self.stop_reason_,
                "best_loss": self.best_loss_,
                "final_loo_loss": self.final_loo_loss_,
                "matrix_norm": float(np.linalg.norm(matrix, "fro")),
                "r2": float(self.diagnostics_["r2"]),
                "adj_r2": float(self.diagnostics_["adj_r2"]),
                "rmse": float(self.diagnostics_["rmse"]),
                "aicc": float(self.diagnostics_["aicc"]),
                "enp": float(self.diagnostics_["enp"]),
            },
        )


__all__ = ["LGGWR", "LGGWRPredictionResult"]
