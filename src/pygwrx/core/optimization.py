# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""One-dimensional optimization for bandwidth selection.

This module provides continuous and discrete search algorithms used to minimize bandwidth-selection objectives safely.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

Number = Union[int, float, np.integer, np.floating]
Objective = Callable[[float], float]


__all__ = [
    "OptimizationResult",
    "GoldenSectionSearch",
    "BrentSearch",
]


@dataclass
class OptimizationResult:
    """Result returned by a one-dimensional optimizer.

    Args:
        value: Best parameter value found.
        score: Objective-function value at ``value``.
        iterations: Number of optimization updates performed.
        converged: Whether the stopping criterion was satisfied and a finite solution
            was found.
        evaluations: Number of unique objective-function evaluations.
        message: Human-readable termination message.

    Notes:
        The first four fields are retained for backward compatibility with the
        original project implementation. ``evaluations`` and ``message`` are
        additive metadata fields.
    """

    value: Union[float, int]
    score: float
    iterations: int
    converged: bool
    evaluations: int = 0
    message: str = ""


def _validate_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean value.")
    return bool(value)


def _validate_positive_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive real scalar, not bool.")

    array = np.asarray(value)
    if array.ndim != 0:
        raise TypeError(f"{name} must be a scalar value.")

    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive real scalar.") from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")

    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _validate_objective(func: object) -> Objective:
    if not callable(func):
        raise TypeError("func must be callable.")
    return func  # type: ignore[return-value]


def _validate_bounds(lower: object, upper: object) -> Tuple[float, float]:
    if isinstance(lower, (bool, np.bool_)) or isinstance(upper, (bool, np.bool_)):
        raise TypeError("lower and upper must be real scalar values, not bool.")

    lower_arr = np.asarray(lower)
    upper_arr = np.asarray(upper)
    if lower_arr.ndim != 0 or upper_arr.ndim != 0:
        raise TypeError("lower and upper must be scalar values.")

    try:
        lower_value = float(lower_arr)
        upper_value = float(upper_arr)
    except (TypeError, ValueError) as exc:
        raise TypeError("lower and upper must be real scalar values.") from exc

    if not np.isfinite(lower_value) or not np.isfinite(upper_value):
        raise ValueError("lower and upper must be finite.")
    if lower_value > upper_value:
        raise ValueError(
            f"lower must be less than or equal to upper; got "
            f"lower={lower_value}, upper={upper_value}."
        )
    return lower_value, upper_value


class _ObjectiveEvaluator:
    """Validate, cache and count scalar objective evaluations."""

    def __init__(self, func: Objective, integer: bool = False):
        self.func = func
        self.integer = integer
        self.cache: Dict[Union[int, float], float] = {}

    @property
    def evaluations(self) -> int:
        return len(self.cache)

    def __call__(self, value: Number) -> float:
        key: Union[int, float]
        if self.integer:
            key = int(round(float(value)))
        else:
            key = float(value)

        if key in self.cache:
            return self.cache[key]

        raw_score = self.func(key)
        score_arr = np.asarray(raw_score)
        if score_arr.ndim != 0:
            raise TypeError(
                "The objective function must return a scalar score; "
                f"got an array with shape {score_arr.shape}."
            )

        try:
            score = float(score_arr)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "The objective function must return a real scalar score."
            ) from exc

        # Non-finite scores do not define a usable optimum. Treat them as
        # invalid candidates while allowing the search to continue.
        if not np.isfinite(score):
            score = np.inf

        self.cache[key] = score
        return score

    def best(self) -> Tuple[Union[int, float], float]:
        if not self.cache:
            return np.nan, np.inf
        return min(self.cache.items(), key=lambda item: (item[1], item[0]))


class GoldenSectionSearch:
    """Golden-section search for one-dimensional minimization.

    Continuous searches use the standard golden-section interval reduction.
    Adaptive bandwidth searches use a discrete integer variant and finish by
    evaluating every integer in the final short bracket. Unlike the original
    implementation, convergence is controlled by ``tol`` rather than by a
    hard-coded constant or by equality of objective values.

    Args:
        tol: Positive convergence tolerance for the search interval.
        max_iter: Maximum number of interval-reduction updates.
        verbose: Whether to print progress information.
    """

    PHI = (1.0 + np.sqrt(5.0)) / 2.0
    RESPHI = (np.sqrt(5.0) - 1.0) / 2.0

    def __init__(
        self,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = True,
    ):
        self.tol = _validate_positive_float(tol, "tol")
        self.max_iter = _validate_positive_int(max_iter, "max_iter")
        self.verbose = _validate_bool(verbose, "verbose")

    def minimize(
        self,
        func: Callable[[float], float],
        lower: float,
        upper: float,
        adaptive: bool = False,
    ) -> OptimizationResult:
        """Minimize a scalar objective on a closed interval.

        Args:
            func: Scalar objective function. Lower scores are better.
            lower: Finite lower search bound.
            upper: Finite upper search bound with ``lower <= upper``.
            adaptive: If ``True``, search only integer nearest-neighbour counts.

        Returns:
            OptimizationResult: Best candidate, objective score, convergence state and metadata.
        """
        objective = _validate_objective(func)
        adaptive_value = _validate_bool(adaptive, "adaptive")
        lower_value, upper_value = _validate_bounds(lower, upper)

        if adaptive_value:
            return self._minimize_integer(
                objective,
                lower_value,
                upper_value,
            )
        return self._minimize_continuous(
            objective,
            lower_value,
            upper_value,
        )

    def _minimize_continuous(
        self,
        func: Objective,
        lower: float,
        upper: float,
    ) -> OptimizationResult:
        evaluator = _ObjectiveEvaluator(func, integer=False)

        if lower == upper:
            score = evaluator(lower)
            finite = np.isfinite(score)
            return OptimizationResult(
                value=lower,
                score=score,
                iterations=0,
                converged=bool(finite),
                evaluations=evaluator.evaluations,
                message=(
                    "The search interval contains a single finite candidate."
                    if finite
                    else "The single search candidate has a non-finite score."
                ),
            )

        a = lower
        b = upper
        c = b - self.RESPHI * (b - a)
        d = a + self.RESPHI * (b - a)

        # Evaluating the endpoints makes constrained endpoint optima observable.
        evaluator(a)
        evaluator(b)
        fc = evaluator(c)
        fd = evaluator(d)

        if self.verbose:
            print("  Starting Golden Section Search")
            print(f"  Search interval: [{a:.6g}, {b:.6g}]")

        iterations = 0
        converged_interval = False

        while iterations < self.max_iter:
            midpoint = 0.5 * (a + b)
            threshold = self.tol * (1.0 + abs(midpoint))
            if (b - a) <= threshold:
                converged_interval = True
                break

            if fc <= fd:
                b = d
                d = c
                fd = fc
                c = b - self.RESPHI * (b - a)
                fc = evaluator(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + self.RESPHI * (b - a)
                fd = evaluator(d)

            iterations += 1

            if self.verbose and iterations % 5 == 0:
                best_x, best_score = evaluator.best()
                print(
                    "  Iteration "
                    f"{iterations}: x={float(best_x):.6g}, "
                    f"f(x)={best_score:.6g}, interval=[{a:.6g}, {b:.6g}]"
                )

        best_value, best_score = evaluator.best()
        finite_solution = np.isfinite(best_score)
        converged = bool(converged_interval and finite_solution)

        if converged:
            message = "Converged because the search interval reached the tolerance."
        elif not finite_solution:
            message = "No finite objective value was found in the search interval."
        else:
            message = "Maximum iterations reached before interval convergence."

        if self.verbose:
            print(f"  Converged: {converged}")
            print(f"  Final interval width: {b - a:.6g}")
            print(f"  Optimal value: {float(best_value):.6g}")
            print(f"  Optimal score: {best_score:.6g}")
            print(f"  Objective evaluations: {evaluator.evaluations}")

        return OptimizationResult(
            value=float(best_value),
            score=float(best_score),
            iterations=iterations,
            converged=converged,
            evaluations=evaluator.evaluations,
            message=message,
        )

    def _minimize_integer(
        self,
        func: Objective,
        lower: float,
        upper: float,
    ) -> OptimizationResult:
        integer_lower = int(np.ceil(lower))
        integer_upper = int(np.floor(upper))

        if integer_lower > integer_upper:
            raise ValueError(
                "The adaptive search interval contains no integer candidate: "
                f"[{lower}, {upper}]."
            )

        evaluator = _ObjectiveEvaluator(func, integer=True)

        if integer_lower == integer_upper:
            score = evaluator(integer_lower)
            finite = np.isfinite(score)
            return OptimizationResult(
                value=integer_lower,
                score=score,
                iterations=0,
                converged=bool(finite),
                evaluations=evaluator.evaluations,
                message=(
                    "The adaptive interval contains one finite integer candidate."
                    if finite
                    else "The only adaptive candidate has a non-finite score."
                ),
            )

        a = integer_lower
        b = integer_upper
        iterations = 0

        if self.verbose:
            print("  Starting Discrete Golden Section Search")
            print(f"  Integer search interval: [{a}, {b}]")

        # Reduce until only a short integer interval remains. Objective-value
        # equality is deliberately not a stopping condition because plateaus are
        # common when continuous proposals map to the same integer bandwidth.
        while (b - a) > 4 and iterations < self.max_iter:
            span = b - a
            c = b - int(np.ceil(self.RESPHI * span))
            d = a + int(np.ceil(self.RESPHI * span))

            c = max(a + 1, min(c, b - 1))
            d = max(a + 1, min(d, b - 1))

            if c >= d:
                c = a + span // 3
                d = b - span // 3
                if c >= d:
                    break

            fc = evaluator(c)
            fd = evaluator(d)

            if fc <= fd:
                b = d
            else:
                a = c

            iterations += 1

            if self.verbose and iterations % 5 == 0:
                best_x, best_score = evaluator.best()
                print(
                    f"  Iteration {iterations}: k={int(best_x)}, "
                    f"f(k)={best_score:.6g}, interval=[{a}, {b}]"
                )

        # Exhaustively inspect the final small bracket and both original bounds.
        final_candidates = set(range(a, b + 1))
        final_candidates.add(integer_lower)
        final_candidates.add(integer_upper)
        for candidate in sorted(final_candidates):
            evaluator(candidate)

        best_value, best_score = evaluator.best()
        finite_solution = np.isfinite(best_score)
        interval_reduced = (b - a) <= 4
        converged = bool(interval_reduced and finite_solution)

        if converged:
            message = "Converged to a short integer bracket and evaluated it exactly."
        elif not finite_solution:
            message = "No finite objective value was found for any evaluated integer."
        else:
            message = "Maximum iterations reached before reducing the integer bracket."

        if self.verbose:
            print(f"  Converged: {converged}")
            print(f"  Final integer interval: [{a}, {b}]")
            print(f"  Optimal k: {int(best_value)}")
            print(f"  Optimal score: {best_score:.6g}")
            print(f"  Objective evaluations: {evaluator.evaluations}")

        return OptimizationResult(
            value=int(best_value),
            score=float(best_score),
            iterations=iterations,
            converged=converged,
            evaluations=evaluator.evaluations,
            message=message,
        )

    @staticmethod
    def auto_bounds(
        coords: np.ndarray,
        adaptive: bool,
        bandwidth_type: str = "gwr",
    ) -> Tuple[float, float]:
        """Derive safe default bandwidth-search bounds from coordinates.

        Args:
            coords: Finite numeric coordinates.
            adaptive: Whether the bandwidth represents an integer neighbour count.
            bandwidth_type: Retained for API compatibility. Both values currently use the same
                robust bounds; unsupported values are rejected rather than ignored.

        Returns:
            (lower, upper) : tuple of float: Valid ordered search bounds.

        Notes:
            Adaptive bounds use ``[20, n]`` for datasets with at least 40 samples
            and a 5%-based lower bound (at least 2) for smaller datasets. Fixed bounds
            are based on positive
            observed pairwise distances, avoiding zero-width bounds for repeated or
            degenerate coordinates.
        """
        adaptive_value = _validate_bool(adaptive, "adaptive")

        if not isinstance(bandwidth_type, str):
            raise TypeError("bandwidth_type must be a string.")
        bandwidth_name = bandwidth_type.strip().lower()
        if bandwidth_name not in {"gwr", "bandwidth"}:
            raise ValueError(
                "bandwidth_type must be either 'gwr' or 'bandwidth'; "
                f"got {bandwidth_type!r}."
            )

        try:
            coords_arr = np.asarray(coords, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("coords must contain numeric values.") from exc

        if coords_arr.ndim != 2:
            raise ValueError(
                "coords must be a two-dimensional array with shape "
                "(n_samples, n_dimensions)."
            )
        if coords_arr.shape[0] < 2:
            raise ValueError("At least two coordinate observations are required.")
        if coords_arr.shape[1] < 1:
            raise ValueError("coords must contain at least one coordinate dimension.")
        if not np.all(np.isfinite(coords_arr)):
            raise ValueError("coords must contain only finite values.")

        n_samples = coords_arr.shape[0]

        if adaptive_value:
            lower = max(2, int(np.ceil(0.05 * n_samples)))
            if n_samples >= 40:
                lower = max(20, lower)
            lower = min(lower, n_samples)
            return float(lower), float(n_samples)

        unique_coords = np.unique(coords_arr, axis=0)
        if unique_coords.shape[0] < 2:
            raise ValueError(
                "Cannot derive fixed-bandwidth bounds because all pairwise "
                "coordinate distances are zero."
            )

        # Use nearest-neighbour distances for a local lower scale without
        # materializing the O(n^2) condensed pairwise-distance vector.
        tree = cKDTree(unique_coords)
        nearest_distances, _ = tree.query(unique_coords, k=2)
        positive_nearest = nearest_distances[:, 1]
        positive_nearest = positive_nearest[positive_nearest > 0.0]

        bbox_min = unique_coords.min(axis=0)
        bbox_max = unique_coords.max(axis=0)
        upper = float(np.linalg.norm(bbox_max - bbox_min))
        lower = float(np.percentile(positive_nearest, 5.0))

        if not np.isfinite(lower) or lower <= 0.0:
            lower = float(np.min(positive_nearest))

        # With only one distinct positive distance, construct a non-degenerate
        # interval while preserving the observed distance as the upper bound.
        if lower >= upper:
            lower = upper / 1000.0
            if lower <= 0.0:
                lower = np.nextafter(0.0, 1.0)

        return lower, upper


class BrentSearch:
    """Brent's bounded method for continuous one-dimensional minimization.

    Args:
        tol: Positive relative/absolute convergence tolerance.
        max_iter: Maximum number of optimization updates.
        verbose: Whether to print progress information.

    Notes:
        Brent's method is a continuous optimizer. Adaptive integer bandwidths
        should normally use ``GoldenSectionSearch(..., adaptive=True)`` or an
        explicit integer post-processing step in the bandwidth selector.
    """

    GOLDEN = 0.3819660112501051
    ZEPS = np.finfo(float).eps * 1e-3

    def __init__(
        self,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = True,
    ):
        self.tol = _validate_positive_float(tol, "tol")
        self.max_iter = _validate_positive_int(max_iter, "max_iter")
        self.verbose = _validate_bool(verbose, "verbose")

    def minimize(
        self,
        func: Callable[[float], float],
        lower: float,
        upper: float,
    ) -> OptimizationResult:
        """Minimize a scalar objective on the closed interval ``[lower, upper]``."""
        objective = _validate_objective(func)
        lower_value, upper_value = _validate_bounds(lower, upper)
        evaluator = _ObjectiveEvaluator(objective, integer=False)

        if lower_value == upper_value:
            score = evaluator(lower_value)
            finite = np.isfinite(score)
            return OptimizationResult(
                value=lower_value,
                score=score,
                iterations=0,
                converged=bool(finite),
                evaluations=evaluator.evaluations,
                message=(
                    "The search interval contains a single finite candidate."
                    if finite
                    else "The single search candidate has a non-finite score."
                ),
            )

        a = lower_value
        b = upper_value

        # Record endpoints so constrained boundary minima can be returned.
        fa = evaluator(a)
        fb = evaluator(b)

        x = w = v = a + self.GOLDEN * (b - a)
        fx = fw = fv = evaluator(x)

        # If the initial interior point is invalid but a finite endpoint exists,
        # probe the midpoint before deciding whether the interior is unusable.
        if not np.isfinite(fx):
            midpoint = 0.5 * (a + b)
            fm = evaluator(midpoint)
            if np.isfinite(fm):
                x = w = v = midpoint
                fx = fw = fv = fm
            elif np.isfinite(fa) or np.isfinite(fb):
                best_value, best_score = evaluator.best()
                return OptimizationResult(
                    value=float(best_value),
                    score=float(best_score),
                    iterations=0,
                    converged=True,
                    evaluations=evaluator.evaluations,
                    message=(
                        "Only a finite boundary candidate was found; returned the "
                        "best constrained endpoint."
                    ),
                )
            else:
                return OptimizationResult(
                    value=float(x),
                    score=np.inf,
                    iterations=0,
                    converged=False,
                    evaluations=evaluator.evaluations,
                    message="No finite objective value was found at initialization.",
                )

        if self.verbose:
            print("  Starting Brent's Method")
            print(f"  Search interval: [{a:.6g}, {b:.6g}]")

        d = 0.0
        e = 0.0
        iterations = 0
        converged_interval = False

        while iterations < self.max_iter:
            midpoint = 0.5 * (a + b)
            tol1 = self.tol * abs(x) + self.ZEPS
            tol2 = 2.0 * tol1

            if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
                converged_interval = True
                break

            use_parabola = (
                abs(e) > tol1
                and np.isfinite(fx)
                and np.isfinite(fw)
                and np.isfinite(fv)
            )

            if use_parabola:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)

                if q > 0.0:
                    p = -p
                q = abs(q)
                previous_e = e
                e = d

                acceptable = (
                    q > 0.0
                    and abs(p) < abs(0.5 * q * previous_e)
                    and p > q * (a - x)
                    and p < q * (b - x)
                )

                if acceptable:
                    d = p / q
                    u = x + d
                    if (u - a) < tol2 or (b - u) < tol2:
                        d = tol1 if midpoint >= x else -tol1
                else:
                    e = a - x if x >= midpoint else b - x
                    d = self.GOLDEN * e
            else:
                e = a - x if x >= midpoint else b - x
                d = self.GOLDEN * e

            step = d if abs(d) >= tol1 else (tol1 if d >= 0.0 else -tol1)
            u = x + step
            fu = evaluator(u)
            iterations += 1

            if fu <= fx:
                if u >= x:
                    a = x
                else:
                    b = x
                v, w, x = w, x, u
                fv, fw, fx = fw, fx, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, w = w, u
                    fv, fw = fw, fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu

            if self.verbose and iterations % 5 == 0:
                best_value, best_score = evaluator.best()
                print(
                    f"  Iteration {iterations}: x={float(best_value):.6g}, "
                    f"f(x)={best_score:.6g}, interval=[{a:.6g}, {b:.6g}]"
                )

        best_value, best_score = evaluator.best()
        finite_solution = np.isfinite(best_score)
        converged = bool(converged_interval and finite_solution)

        if converged:
            message = "Converged because the bounded Brent criterion was satisfied."
        elif not finite_solution:
            message = "No finite objective value was found in the search interval."
        else:
            message = "Maximum iterations reached before Brent convergence."

        if self.verbose:
            print(f"  Converged: {converged}")
            print(f"  Final interval width: {b - a:.6g}")
            print(f"  Optimal value: {float(best_value):.6g}")
            print(f"  Optimal score: {best_score:.6g}")
            print(f"  Objective evaluations: {evaluator.evaluations}")

        return OptimizationResult(
            value=float(best_value),
            score=float(best_score),
            iterations=iterations,
            converged=converged,
            evaluations=evaluator.evaluations,
            message=message,
        )
