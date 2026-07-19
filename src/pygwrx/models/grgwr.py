# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Geo-regime geographically weighted regression.

GR-GWR discovers spatially connected regimes from an initial GWR coefficient
field and fits smoothly varying local coefficients inside each regime while
allowing discontinuities across regime boundaries.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from pygwrx._optional import import_optional_dependency
from pygwrx.core._summary import format_summary
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.utils import add_intercept, validate_coords

ArrayLike = Union[np.ndarray, pd.DataFrame]
VectorLike = Union[np.ndarray, pd.Series]
BandwidthLike = Union[int, float]


@dataclass(frozen=True)
class GRGWRPredictionResult:
    """Detailed GR-GWR predictions at evaluation locations."""

    predictions: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    regimes: np.ndarray
    coords: np.ndarray
    feature_names: Tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        """Return predictions, regimes and local parameters as a DataFrame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "prediction": self.predictions,
            "regime": self.regimes,
            "intercept": self.intercepts,
        }
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coefficients[:, index]
        return pd.DataFrame(data)


@dataclass(frozen=True)
class _RegimeFit:
    local_parameters: np.ndarray
    fitted_values: np.ndarray
    hat_matrix: np.ndarray
    coefficients_by_regime: Tuple[np.ndarray, ...]


class GRGWR:
    r"""Geo-Regime Geographically Weighted Regression.

    GR-GWR models a piecewise-smooth coefficient field.  An initial full-domain
    GWR provides local slope features.  Spatially constrained agglomerative
    clustering produces connected initial regimes, and a sequential ICM update
    refines labels under

    .. math::

        L(z)=\sum_i(y_i-x_i^T\beta_i^{(z_i)})^2+\lambda B(z),

    where :math:`B(z)` counts undirected neighbouring pairs with different
    regime labels.  Every accepted ICM move preserves the source regime's
    connectivity and attaches the point to an adjacent target regime.  A full
    refit is accepted only when the reported objective does not increase.

    Args:
        n_regimes: Requested number of regimes.
        bandwidth: Positive fixed distance or one-based adaptive neighbour count.
        kernel: ``"bisquare"``, ``"gaussian"`` or ``"exponential"``.
        lambda_boundary: Non-negative boundary-length penalty.
        max_iter: Maximum ICM sweeps.
        tol: Objective tolerance.
        spatial_constraint_weight: :math:`\gamma` in ``[0, 1]``.  Clustering
            uses ``sqrt(1-gamma)`` times standardized slope coefficients and
            ``sqrt(gamma)`` times normalized coordinates, so the endpoints are
            exactly coefficient-only and coordinate-only.
        fit_intercept: Add a local intercept.  A legacy leading all-ones column
            is detected and removed.
        n_neighbors: k for the symmetric kNN adjacency graph.  A minimum
            spanning tree is added so the graph is connected.
        min_regime_size: Minimum members per regime.  ``None`` uses the number
            of design parameters plus two.
        enforce_connectivity: Preserve connected regimes during ICM.
        random_state: Deterministic clustering and ICM order seed.
        verbose: Print fitting progress.

    Notes:
        The reported AICc and ENP are **conditional on the discovered regime
        labels**.  They measure the final piecewise local smoother and do not
        include the full discrete search complexity of regime discovery.
    """

    _KERNELS = {"bisquare", "gaussian", "exponential"}

    def __init__(
        self,
        n_regimes: int = 3,
        bandwidth: BandwidthLike = 20,
        kernel: str = "bisquare",
        lambda_boundary: float = 1.0,
        max_iter: int = 10,
        tol: float = 1e-4,
        spatial_constraint_weight: float = 0.5,
        fit_intercept: bool = True,
        verbose: bool = False,
        *,
        n_neighbors: int = 8,
        min_regime_size: Optional[int] = None,
        enforce_connectivity: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_regimes = self._positive_int(n_regimes, "n_regimes")
        self.bandwidth = self._validate_bandwidth(bandwidth)
        self.kernel = self._choice(kernel, "kernel", self._KERNELS)
        self.lambda_boundary = self._nonnegative_float(
            lambda_boundary, "lambda_boundary"
        )
        self.max_iter = self._nonnegative_int(max_iter, "max_iter")
        self.tol = self._positive_float(tol, "tol")
        self.spatial_constraint_weight = self._unit_float(
            spatial_constraint_weight, "spatial_constraint_weight"
        )
        self.fit_intercept = self._boolean(fit_intercept, "fit_intercept")
        self.verbose = self._boolean(verbose, "verbose")
        self.n_neighbors = self._positive_int(n_neighbors, "n_neighbors")
        self.min_regime_size = (
            None
            if min_regime_size is None
            else self._positive_int(min_regime_size, "min_regime_size")
        )
        self.enforce_connectivity = self._boolean(
            enforce_connectivity, "enforce_connectivity"
        )
        self.random_state = random_state
        self._reset_fit_state()

    # ------------------------------------------------------------------
    # Validation and input handling
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

    @classmethod
    def _unit_float(cls, value: float, name: str) -> float:
        result = cls._nonnegative_float(value, name)
        if result > 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
        return result

    @staticmethod
    def _choice(value: str, name: str, choices: set[str]) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        result = value.strip().lower()
        if result not in choices:
            raise ValueError(f"{name} must be one of {sorted(choices)}.")
        return result

    @classmethod
    def _validate_bandwidth(cls, value: BandwidthLike) -> BandwidthLike:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("bandwidth must be a positive integer or float.")
        if isinstance(value, Integral):
            return cls._positive_int(value, "bandwidth")
        return cls._positive_float(value, "bandwidth")

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
        names = (
            tuple(str(column) for column in X.columns)
            if isinstance(X, pd.DataFrame)
            else tuple(f"x{index}" for index in range(array.shape[1]))
        )
        self._legacy_intercept_input_ = False
        if self.fit_intercept and np.allclose(array[:, 0], 1.0):
            self._legacy_intercept_input_ = True
            array = array[:, 1:]
            names = names[1:]
            if array.shape[1] == 0:
                raise ValueError("X must contain a non-intercept predictor.")
            import warnings

            warnings.warn(
                "A leading all-ones column was removed because fit_intercept=True.",
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
        if names is not None and names != self.feature_names_:
            raise ValueError(
                "Prediction DataFrame columns must match training columns in order. "
                f"Expected {list(self.feature_names_)}, got {list(names)}."
            )
        return array

    def _reset_fit_state(self) -> None:
        self.X_: Optional[np.ndarray] = None
        self._Xd: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.coords_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.feature_names_: Tuple[str, ...] = ()
        self.n_features_in_: Optional[int] = None
        self.distance_matrix_: Optional[np.ndarray] = None
        self.adjacency_matrix_: Optional[csr_matrix] = None
        self.adjacency_: Tuple[np.ndarray, ...] = ()
        self.edges_: Tuple[Tuple[int, int], ...] = ()
        self.global_coef_: Optional[np.ndarray] = None
        self.clustering_features_: Optional[np.ndarray] = None
        self.regimes_: Optional[np.ndarray] = None
        self.n_regimes_actual_: Optional[int] = None
        self.regime_sizes_: Optional[np.ndarray] = None
        self.regime_component_counts_: Optional[np.ndarray] = None
        self.local_parameters_: Optional[np.ndarray] = None
        self.coefficients_: Optional[Tuple[np.ndarray, ...]] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.hat_matrix_: Optional[np.ndarray] = None
        self.regime_boundaries_: Tuple[Tuple[int, int], ...] = ()
        self.objective_history_: list[float] = []
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.stop_reason_: Optional[str] = None
        self.diagnostics_: Optional[Dict[str, Any]] = None
        self.search_results_: Optional[pd.DataFrame] = None
        self.selection_criterion_: Optional[str] = None
        self._legacy_intercept_input_: bool = False
        self._min_regime_size_: Optional[int] = None
        self._is_fitted = False

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("GRGWR is not fitted. Call fit() first.")

    # ------------------------------------------------------------------
    # Spatial graph and local WLS
    # ------------------------------------------------------------------
    def _build_graph(self, coords: np.ndarray) -> None:
        n = coords.shape[0]
        self.distance_matrix_ = cdist(coords, coords)
        k = min(self.n_neighbors, n - 1)
        graph = np.zeros((n, n), dtype=bool)
        for i in range(n):
            neighbours = np.argsort(self.distance_matrix_[i])[1 : k + 1]
            graph[i, neighbours] = True
        graph |= graph.T

        # Add a Euclidean minimum spanning tree so spatially constrained
        # clustering always receives one connected graph.
        mst = minimum_spanning_tree(self.distance_matrix_).tocoo()
        for i, j in zip(mst.row, mst.col):
            graph[int(i), int(j)] = True
            graph[int(j), int(i)] = True
        np.fill_diagonal(graph, False)
        self.adjacency_matrix_ = csr_matrix(graph.astype(float))
        self.adjacency_ = tuple(np.flatnonzero(graph[i]) for i in range(n))
        self.edges_ = tuple(
            (int(i), int(j)) for i in range(n) for j in self.adjacency_[i] if i < j
        )

    def _weights(self, distances: np.ndarray) -> np.ndarray:
        if isinstance(self.bandwidth, Integral):
            k = min(int(self.bandwidth), distances.size)
            bandwidth = float(np.partition(distances, k - 1)[k - 1])
            if bandwidth <= 1e-12:
                positive = distances[distances > 1e-12]
                bandwidth = float(np.min(positive)) if positive.size else 1.0
            bandwidth *= 1.0000001
        else:
            bandwidth = float(self.bandwidth)
        ratio = distances / bandwidth
        if self.kernel == "bisquare":
            return np.where(ratio < 1.0, (1.0 - ratio**2) ** 2, 0.0)
        if self.kernel == "gaussian":
            return np.exp(-0.5 * ratio**2)
        return np.exp(-ratio)

    @staticmethod
    def _solve_local(
        X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xw = X * weights[:, None]
        M = Xw.T @ X
        p = X.shape[1]
        try:
            C = np.linalg.solve(M, Xw.T)
            beta = C @ y
            if np.all(np.isfinite(beta)):
                return beta, C
        except np.linalg.LinAlgError:
            pass
        ridge = 1e-6 * (np.trace(M) / max(p, 1) + 1e-12) + 1e-12
        try:
            C = np.linalg.solve(M + ridge * np.eye(p), Xw.T)
            beta = C @ y
            if np.all(np.isfinite(beta)):
                return beta, C
        except np.linalg.LinAlgError:
            pass
        C = np.linalg.pinv(M) @ Xw.T
        return C @ y, C

    def _fit_global_gwr(self) -> np.ndarray:
        if self._Xd is None or self.y_ is None or self.distance_matrix_ is None:
            raise RuntimeError("Training state is incomplete.")
        n, p = self._Xd.shape
        parameters = np.zeros((n, p))
        for i in range(n):
            parameters[i], _ = self._solve_local(
                self._Xd, self.y_, self._weights(self.distance_matrix_[i])
            )
        return parameters

    # ------------------------------------------------------------------
    # Initial connected regimes
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_standardize(values: np.ndarray) -> np.ndarray:
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0, ddof=0)
        scale = np.where(scale <= 1e-12, 1.0, scale)
        return (values - mean) / scale

    def _clustering_features(
        self, local_parameters: np.ndarray, coords: np.ndarray
    ) -> np.ndarray:
        slopes = local_parameters[:, 1:] if self.fit_intercept else local_parameters
        slopes_scaled = self._safe_standardize(slopes)
        coordinate_range = np.ptp(coords, axis=0)
        coordinate_range = np.where(coordinate_range <= 1e-12, 1.0, coordinate_range)
        coords_scaled = (coords - np.min(coords, axis=0)) / coordinate_range
        gamma = self.spatial_constraint_weight
        if gamma == 0.0:
            return slopes_scaled
        if gamma == 1.0:
            return coords_scaled
        return np.hstack(
            [np.sqrt(1.0 - gamma) * slopes_scaled, np.sqrt(gamma) * coords_scaled]
        )

    @staticmethod
    def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
        unique = np.unique(labels)
        mapping = {int(label): index for index, label in enumerate(unique)}
        return np.asarray([mapping[int(label)] for label in labels], dtype=int)

    def _initial_regimes(self, features: np.ndarray) -> np.ndarray:
        if self.adjacency_matrix_ is None:
            raise RuntimeError("Adjacency graph is unavailable.")
        n = features.shape[0]
        feasible = max(1, n // self._min_regime_size_)
        n_clusters = min(self.n_regimes, feasible)
        if n_clusters < self.n_regimes:
            import warnings

            warnings.warn(
                f"n_regimes reduced from {self.n_regimes} to {n_clusters} because "
                "of min_regime_size.",
                UserWarning,
                stacklevel=3,
            )
        if n_clusters == 1:
            return np.zeros(n, dtype=int)
        cluster = import_optional_dependency(
            "sklearn.cluster", extra="ml", purpose="GRGWR regime initialization"
        )
        model = cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            connectivity=self.adjacency_matrix_,
            compute_full_tree=True,
        )
        labels = model.fit_predict(features)
        labels = self._merge_small_regimes(labels, features)
        return self._relabel_contiguous(labels)

    def _merge_small_regimes(
        self, labels: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        labels = self._relabel_contiguous(labels.copy())
        while True:
            counts = np.bincount(labels)
            small = np.flatnonzero(counts < self._min_regime_size_)
            if small.size == 0 or counts.size == 1:
                return labels
            source = int(small[np.argmin(counts[small])])
            source_nodes = np.flatnonzero(labels == source)
            adjacent_counts: Dict[int, int] = {}
            for node in source_nodes:
                for neighbour in self.adjacency_[node]:
                    target = int(labels[neighbour])
                    if target != source:
                        adjacent_counts[target] = adjacent_counts.get(target, 0) + 1
            if adjacent_counts:
                max_edges = max(adjacent_counts.values())
                candidates = [
                    label
                    for label, value in adjacent_counts.items()
                    if value == max_edges
                ]
            else:
                candidates = [label for label in range(counts.size) if label != source]
            source_center = np.mean(features[source_nodes], axis=0)
            target = min(
                candidates,
                key=lambda label: float(
                    np.linalg.norm(
                        source_center - np.mean(features[labels == label], axis=0)
                    )
                ),
            )
            labels[source_nodes] = int(target)
            labels = self._relabel_contiguous(labels)

    # ------------------------------------------------------------------
    # Regime fit, objective and ICM
    # ------------------------------------------------------------------
    def _fit_for_labels(self, labels: np.ndarray) -> _RegimeFit:
        if self._Xd is None or self.y_ is None or self.distance_matrix_ is None:
            raise RuntimeError("Training state is incomplete.")
        n, p = self._Xd.shape
        local_parameters = np.zeros((n, p))
        fitted = np.zeros(n)
        hat = np.zeros((n, n))
        blocks = []
        for regime in range(int(np.max(labels)) + 1):
            indices = np.flatnonzero(labels == regime)
            block = np.zeros((indices.size, p))
            X_r = self._Xd[indices]
            y_r = self.y_[indices]
            for local_index, global_index in enumerate(indices):
                distances = self.distance_matrix_[global_index, indices]
                beta, C = self._solve_local(X_r, y_r, self._weights(distances))
                local_parameters[global_index] = beta
                block[local_index] = beta
                fitted[global_index] = self._Xd[global_index] @ beta
                hat[global_index, indices] = self._Xd[global_index] @ C
            blocks.append(block)
        return _RegimeFit(
            local_parameters=local_parameters,
            fitted_values=fitted,
            hat_matrix=hat,
            coefficients_by_regime=tuple(blocks),
        )

    def _boundary_count(self, labels: np.ndarray) -> int:
        return int(sum(labels[i] != labels[j] for i, j in self.edges_))

    def _objective(self, labels: np.ndarray, fit: _RegimeFit) -> float:
        rss = float(np.sum((self.y_ - fit.fitted_values) ** 2))
        return rss + self.lambda_boundary * self._boundary_count(labels)

    def _candidate_error(self, node: int, regime: int, labels: np.ndarray) -> float:
        """Return the leave-one-out local WLS error for one candidate regime.

        The fit is recalibrated from the *current* labels at the moment the node
        is visited.  Excluding the focal observation prevents a candidate from
        winning merely through its own unit kernel weight.
        """
        if self._Xd is None or self.y_ is None or self.distance_matrix_ is None:
            raise RuntimeError("Training state is incomplete.")
        indices = np.flatnonzero(labels == regime)
        indices = indices[indices != node]
        if indices.size < self._Xd.shape[1]:
            return float("inf")
        beta, _ = self._solve_local(
            self._Xd[indices],
            self.y_[indices],
            self._weights(self.distance_matrix_[node, indices]),
        )
        prediction = float(self._Xd[node] @ beta)
        return float((self.y_[node] - prediction) ** 2)

    def _can_remove_from_source(self, node: int, labels: np.ndarray) -> bool:
        """Check minimum-size and optional connectivity constraints."""
        source = int(labels[node])
        members = np.flatnonzero(labels == source)
        if members.size - 1 < self._min_regime_size_:
            return False
        if not self.enforce_connectivity:
            return True
        remaining = members[members != node]
        if remaining.size <= 1:
            return True
        allowed = set(int(value) for value in remaining)
        visited = {int(remaining[0])}
        stack = [int(remaining[0])]
        while stack:
            current = stack.pop()
            for neighbour in self.adjacency_[current]:
                neighbour_int = int(neighbour)
                if neighbour_int in allowed and neighbour_int not in visited:
                    visited.add(neighbour_int)
                    stack.append(neighbour_int)
        return len(visited) == remaining.size

    def _icm_sweep(self, labels: np.ndarray, iteration: int) -> Tuple[np.ndarray, int]:
        """Run one sequential ICM sweep using current-label local fits."""
        updated = labels.copy()
        rng_seed = (
            0 if self.random_state is None else int(self.random_state)
        ) + iteration
        order = np.random.default_rng(rng_seed).permutation(labels.size)
        changed = 0
        for node_value in order:
            node = int(node_value)
            current = int(updated[node])
            if not self._can_remove_from_source(node, updated):
                continue
            neighbour_labels = set(int(updated[j]) for j in self.adjacency_[node])
            candidates = sorted(neighbour_labels | {current})
            costs: Dict[int, float] = {}
            for candidate in candidates:
                disagreement = int(np.sum(updated[self.adjacency_[node]] != candidate))
                costs[candidate] = (
                    self._candidate_error(node, candidate, updated)
                    + self.lambda_boundary * disagreement
                )
            best = min(costs, key=costs.get)
            if best != current and costs[best] < costs[current] - self.tol:
                updated[node] = int(best)
                changed += 1
        return updated, changed

    def _component_count(self, labels: np.ndarray, regime: int) -> int:
        members = set(int(value) for value in np.flatnonzero(labels == regime))
        count = 0
        while members:
            count += 1
            start = members.pop()
            stack = [start]
            while stack:
                current = stack.pop()
                for neighbour in self.adjacency_[current]:
                    neighbour_int = int(neighbour)
                    if neighbour_int in members and labels[neighbour_int] == regime:
                        members.remove(neighbour_int)
                        stack.append(neighbour_int)
        return count

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: VectorLike, coords: ArrayLike) -> "GRGWR":
        """Fit GR-GWR and return ``self``."""
        self._reset_fit_state()
        try:
            X_raw, names = self._coerce_X_fit(X)
            y_arr = self._numeric_y(y)
            coords_arr = np.asarray(validate_coords(coords), dtype=float)
            if X_raw.shape[0] != y_arr.size or coords_arr.shape[0] != y_arr.size:
                raise ValueError(
                    "X, y and coords must contain the same number of rows."
                )
            self.X_ = X_raw.copy()
            self._Xd = add_intercept(X_raw) if self.fit_intercept else X_raw.copy()
            self.y_ = y_arr.copy()
            self.coords_ = coords_arr.copy()
            self.feature_names_ = names
            self.feature_names_in_ = np.asarray(names, dtype=object)
            self.n_features_in_ = X_raw.shape[1]
            p = self._Xd.shape[1]
            self._min_regime_size_ = (
                p + 2 if self.min_regime_size is None else self.min_regime_size
            )
            if isinstance(self.bandwidth, Integral) and int(self.bandwidth) < p + 1:
                raise ValueError(
                    "Adaptive bandwidth must be at least the number of design "
                    "parameters plus one."
                )
            if y_arr.size < self._min_regime_size_:
                raise ValueError("The sample is smaller than min_regime_size.")

            self._build_graph(coords_arr)
            self.global_coef_ = self._fit_global_gwr()
            self.clustering_features_ = self._clustering_features(
                self.global_coef_, coords_arr
            )
            labels = self._initial_regimes(self.clustering_features_)
            labels = self._relabel_contiguous(labels)

            current_fit = self._fit_for_labels(labels)
            current_objective = self._objective(labels, current_fit)
            self.objective_history_ = [float(current_objective)]
            converged = False
            stop_reason = "max_iter"
            accepted_iterations = 0

            for iteration in range(self.max_iter):
                proposed, changed = self._icm_sweep(labels, iteration)
                if changed == 0:
                    converged = True
                    stop_reason = "labels_stable"
                    break
                proposed_fit = self._fit_for_labels(proposed)
                proposed_objective = self._objective(proposed, proposed_fit)
                if proposed_objective > current_objective + self.tol:
                    converged = True
                    stop_reason = "objective_guard"
                    break
                improvement = current_objective - proposed_objective
                labels = proposed
                current_fit = proposed_fit
                current_objective = proposed_objective
                self.objective_history_.append(float(current_objective))
                accepted_iterations += 1
                if improvement < self.tol:
                    converged = True
                    stop_reason = "objective_tolerance"
                    break

            self.regimes_ = self._relabel_contiguous(labels)
            # Relabelling cannot change the fit when labels are already contiguous,
            # but refit explicitly so every stored block follows final label order.
            final_fit = self._fit_for_labels(self.regimes_)
            self.n_regimes_actual_ = int(np.max(self.regimes_)) + 1
            self.regime_sizes_ = np.bincount(
                self.regimes_, minlength=self.n_regimes_actual_
            )
            self.regime_component_counts_ = np.asarray(
                [
                    self._component_count(self.regimes_, regime)
                    for regime in range(self.n_regimes_actual_)
                ],
                dtype=int,
            )
            self.local_parameters_ = final_fit.local_parameters
            self.coefficients_ = final_fit.coefficients_by_regime
            self.fitted_values_ = final_fit.fitted_values
            self.hat_matrix_ = final_fit.hat_matrix
            self.residuals_ = self.y_ - self.fitted_values_
            self.regime_boundaries_ = tuple(
                (i, j) for i, j in self.edges_ if self.regimes_[i] != self.regimes_[j]
            )
            self.n_iter_ = accepted_iterations
            self.converged_ = converged
            self.stop_reason_ = stop_reason
            self._finalise_parameters()
            self._compute_diagnostics()
            self._is_fitted = True
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def _finalise_parameters(self) -> None:
        if self.local_parameters_ is None:
            raise RuntimeError("Local parameters are unavailable.")
        if self.fit_intercept:
            self.intercept_ = self.local_parameters_[:, 0].copy()
            self.coef_ = self.local_parameters_[:, 1:].copy()
        else:
            self.intercept_ = np.zeros(self.local_parameters_.shape[0])
            self.coef_ = self.local_parameters_.copy()

    def _compute_diagnostics(self) -> None:
        if self.hat_matrix_ is None or self.fitted_values_ is None:
            raise RuntimeError("Fit results are incomplete.")
        base = compute_diagnostics(
            self.y_,
            self.fitted_values_,
            hat_matrix=self.hat_matrix_,
            compute_gwr_stats=True,
        )
        regime_stats = []
        for regime in range(self.n_regimes_actual_):
            mask = self.regimes_ == regime
            residual = self.residuals_[mask]
            centered = self.y_[mask] - np.mean(self.y_[mask])
            tss = float(np.dot(centered, centered))
            rss = float(np.dot(residual, residual))
            regime_stats.append(
                {
                    "regime": regime,
                    "n_samples": int(np.sum(mask)),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mae": float(np.mean(np.abs(residual))),
                    "r2": 1.0 - rss / tss if tss > 0.0 else float("nan"),
                    "n_components": int(self.regime_component_counts_[regime]),
                }
            )
        self.diagnostics_ = dict(base)
        self.diagnostics_.update(
            {
                "rss": float(np.sum(self.residuals_**2)),
                "mae": float(np.mean(np.abs(self.residuals_))),
                "n_regimes": self.n_regimes_actual_,
                "n_boundaries": len(self.regime_boundaries_),
                "conditional_aic": float(base["aic"]),
                "conditional_aicc": float(base["aicc"]),
                "conditional_bic": float(base["bic"]),
                "conditional_enp": float(base["trace_S"]),
                "conditional_enp_v2": float(base["enp_v2"]),
                "boundary_penalty": float(
                    self.lambda_boundary * len(self.regime_boundaries_)
                ),
                "objective": float(
                    np.sum(self.residuals_**2)
                    + self.lambda_boundary * len(self.regime_boundaries_)
                ),
                "regime_stats": regime_stats,
            }
        )

    # ------------------------------------------------------------------
    # Prediction and reporting
    # ------------------------------------------------------------------
    def _assigned_regime(self, distances: np.ndarray) -> int:
        k = min(self.n_neighbors, distances.size)
        neighbours = np.argsort(distances)[:k]
        labels = self.regimes_[neighbours]
        inverse = 1.0 / np.maximum(distances[neighbours], 1e-12)
        scores = np.bincount(labels, weights=inverse, minlength=self.n_regimes_actual_)
        return int(np.argmax(scores))

    def predict_result(self, X: ArrayLike, coords: ArrayLike) -> GRGWRPredictionResult:
        """Assign query regimes and recalibrate local WLS coefficients."""
        self._require_fitted()
        X_raw = self._coerce_X_predict(X)
        X_design = add_intercept(X_raw) if self.fit_intercept else X_raw.copy()
        coords_arr = np.asarray(validate_coords(coords), dtype=float)
        if X_raw.shape[0] != coords_arr.shape[0]:
            raise ValueError("X and coords must contain the same rows.")
        distances = cdist(coords_arr, self.coords_)
        n_query = X_raw.shape[0]
        parameters = np.zeros((n_query, self._Xd.shape[1]))
        regimes = np.zeros(n_query, dtype=int)
        global_beta = np.linalg.lstsq(self._Xd, self.y_, rcond=None)[0]
        for i in range(n_query):
            regime = self._assigned_regime(distances[i])
            regimes[i] = regime
            indices = np.flatnonzero(self.regimes_ == regime)
            if indices.size < self._Xd.shape[1]:
                parameters[i] = global_beta
                continue
            beta, _ = self._solve_local(
                self._Xd[indices],
                self.y_[indices],
                self._weights(distances[i, indices]),
            )
            parameters[i] = beta if np.all(np.isfinite(beta)) else global_beta
        predictions = np.einsum("ij,ij->i", X_design, parameters)
        if self.fit_intercept:
            intercepts = parameters[:, 0]
            coefficients = parameters[:, 1:]
        else:
            intercepts = np.zeros(n_query)
            coefficients = parameters
        return GRGWRPredictionResult(
            predictions=predictions,
            coefficients=coefficients,
            intercepts=intercepts,
            regimes=regimes,
            coords=coords_arr.copy(),
            feature_names=self.feature_names_,
        )

    def predict(self, X: ArrayLike, coords: ArrayLike) -> np.ndarray:
        """Return direct GR-GWR predictions at query locations."""
        return self.predict_result(X, coords).predictions

    def results_frame(self) -> pd.DataFrame:
        """Return training regimes, local parameters and fitted values."""
        self._require_fitted()
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords_[:, 0],
            "coord_1": self.coords_[:, 1],
            "regime": self.regimes_,
            "fitted": self.fitted_values_,
            "residual": self.residuals_,
            "intercept": self.intercept_,
        }
        for index, name in enumerate(self.feature_names_):
            data[f"coef_{name}"] = self.coef_[:, index]
        return pd.DataFrame(data)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`results_frame`."""
        return self.results_frame()

    @classmethod
    def select_parameters(
        cls,
        X: ArrayLike,
        y: VectorLike,
        coords: ArrayLike,
        *,
        n_regimes_grid: Tuple[int, ...] = (2, 3),
        bandwidth_grid: Tuple[BandwidthLike, ...] = (20, 30),
        lambda_boundary_grid: Tuple[float, ...] = (0.0, 1.0),
        spatial_constraint_grid: Tuple[float, ...] = (0.25, 0.5, 0.75),
        criterion: str = "conditional_aicc",
        cv_folds: int = 5,
        random_state: Optional[int] = 42,
        **model_kwargs: Any,
    ) -> Tuple["GRGWR", pd.DataFrame]:
        """Select a modest GR-GWR parameter grid and fit the best model.

        ``criterion="conditional_aicc"`` compares final smoothers conditional
        on their discovered labels. ``criterion="spatial_cv"`` forms compact
        coordinate clusters and reports mean held-out squared error. The search
        is intentionally explicit and exhaustive; users should keep the grids
        small because every candidate contains a regime-discovery fit.

        Returns:
            ``(best_model, search_table)`` sorted by ascending score.
        """
        criterion_key = cls._choice(
            criterion, "criterion", {"conditional_aicc", "spatial_cv"}
        )
        grids = (
            tuple(n_regimes_grid),
            tuple(bandwidth_grid),
            tuple(lambda_boundary_grid),
            tuple(spatial_constraint_grid),
        )
        if any(len(grid) == 0 for grid in grids):
            raise ValueError("All parameter grids must contain at least one value.")
        forbidden = {
            "n_regimes",
            "bandwidth",
            "lambda_boundary",
            "spatial_constraint_weight",
            "random_state",
        } & set(model_kwargs)
        if forbidden:
            raise ValueError(
                "Searched parameters must be supplied through their grid arguments: "
                f"{sorted(forbidden)}."
            )

        y_array = cls._numeric_y(y)
        coords_array = np.asarray(validate_coords(coords), dtype=float)
        X_rows = X.shape[0] if hasattr(X, "shape") else np.asarray(X).shape[0]
        if X_rows != y_array.size or coords_array.shape[0] != y_array.size:
            raise ValueError("X, y and coords must contain the same number of rows.")

        fold_labels: Optional[np.ndarray] = None
        if criterion_key == "spatial_cv":
            folds = cls._positive_int(cv_folds, "cv_folds")
            folds = min(folds, y_array.size)
            if folds < 2:
                raise ValueError("spatial_cv requires at least two folds.")
            cluster = import_optional_dependency(
                "sklearn.cluster", extra="ml", purpose="GRGWR spatial cross-validation"
            )
            fold_labels = cluster.KMeans(
                n_clusters=folds,
                n_init=10,
                random_state=random_state,
            ).fit_predict(coords_array)

        def subset_rows(value: Any, indices: np.ndarray) -> Any:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                return value.iloc[indices]
            return np.asarray(value)[indices]

        records = []
        candidate_parameters = list(product(*grids))
        for candidate_id, (n_regimes, bandwidth, boundary, gamma) in enumerate(
            candidate_parameters
        ):
            parameters = {
                "n_regimes": n_regimes,
                "bandwidth": bandwidth,
                "lambda_boundary": boundary,
                "spatial_constraint_weight": gamma,
                "random_state": random_state,
                **model_kwargs,
            }
            if criterion_key == "conditional_aicc":
                candidate = cls(**parameters).fit(X, y, coords)
                score = float(candidate.diagnostics_["conditional_aicc"])
                fold_scores: Tuple[float, ...] = ()
            else:
                scores = []
                for fold in range(int(np.max(fold_labels)) + 1):
                    test_index = np.flatnonzero(fold_labels == fold)
                    train_index = np.flatnonzero(fold_labels != fold)
                    candidate = cls(**parameters).fit(
                        subset_rows(X, train_index),
                        subset_rows(y, train_index),
                        coords_array[train_index],
                    )
                    prediction = candidate.predict(
                        subset_rows(X, test_index), coords_array[test_index]
                    )
                    scores.append(
                        float(np.mean((y_array[test_index] - prediction) ** 2))
                    )
                fold_scores = tuple(scores)
                score = float(np.mean(scores))
            records.append(
                {
                    "candidate_id": candidate_id,
                    "n_regimes": int(n_regimes),
                    "bandwidth": bandwidth,
                    "lambda_boundary": float(boundary),
                    "spatial_constraint_weight": float(gamma),
                    "criterion": criterion_key,
                    "score": score,
                    "fold_scores": fold_scores,
                }
            )

        table = pd.DataFrame(records).sort_values(
            ["score", "n_regimes", "lambda_boundary"], ignore_index=True
        )
        best = table.iloc[0]
        best_record = records[int(best["candidate_id"])]
        best_model = cls(
            n_regimes=int(best_record["n_regimes"]),
            bandwidth=best_record["bandwidth"],
            lambda_boundary=float(best_record["lambda_boundary"]),
            spatial_constraint_weight=float(best_record["spatial_constraint_weight"]),
            random_state=random_state,
            **model_kwargs,
        ).fit(X, y, coords)
        best_model.search_results_ = table.copy()
        best_model.selection_criterion_ = criterion_key
        return best_model, table

    def summary(self) -> str:
        """Return a plain-text fitted-model summary."""
        self._require_fitted()
        return format_summary(
            "GR-GWR Summary",
            {
                "model": "GR-GWR",
                "n_samples": int(self.y_.size),
                "n_features": int(self.n_features_in_),
                "n_regimes_requested": int(self.n_regimes),
                "n_regimes_actual": int(self.n_regimes_actual_),
                "regime_sizes": tuple(int(value) for value in self.regime_sizes_),
                "component_counts": tuple(
                    int(value) for value in self.regime_component_counts_
                ),
                "bandwidth": self.bandwidth,
                "kernel": self.kernel,
                "lambda_boundary": self.lambda_boundary,
                "n_neighbors": self.n_neighbors,
                "min_regime_size": self._min_regime_size_,
                "n_iterations": self.n_iter_,
                "converged": self.converged_,
                "stop_reason": self.stop_reason_,
                "objective_history": tuple(self.objective_history_),
                "r2": float(self.diagnostics_["r2"]),
                "adj_r2": float(self.diagnostics_["adj_r2"]),
                "rmse": float(self.diagnostics_["rmse"]),
                "conditional_aicc": float(self.diagnostics_["conditional_aicc"]),
                "conditional_enp": float(self.diagnostics_["conditional_enp"]),
                "n_boundaries": len(self.regime_boundaries_),
            },
        )


__all__ = ["GRGWR", "GRGWRPredictionResult"]
