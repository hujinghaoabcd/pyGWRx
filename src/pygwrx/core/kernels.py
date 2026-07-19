# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Spatial kernel functions for geographically weighted models.

The functions in this module convert distances into fixed- or adaptive-bandwidth observation weights.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Callable, Union

import numpy as np

KernelCallable = Callable[[np.ndarray, float], np.ndarray]
KernelLike = Union[str, KernelCallable]


__all__ = [
    "gaussian_kernel",
    "bisquare_kernel",
    "exponential_kernel",
    "tricube_kernel",
    "boxcar_kernel",
    "get_kernel_function",
    "KERNEL_FUNCTIONS",
]


def _validate_distances(distances: np.ndarray) -> np.ndarray:
    """Validate distances and return a floating-point NumPy array."""
    try:
        distances_arr = np.asarray(distances, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("distances must be numeric array-like data.") from exc

    if distances_arr.ndim == 0:
        distances_arr = distances_arr.reshape(1)

    if not np.all(np.isfinite(distances_arr)):
        raise ValueError("distances must contain only finite values.")

    if np.any(distances_arr < 0):
        raise ValueError("distances must be non-negative.")

    return distances_arr


def _validate_bandwidth(bandwidth: float) -> float:
    """Validate a scalar bandwidth and return it as a float."""
    if isinstance(bandwidth, (bool, np.bool_)):
        raise TypeError("bandwidth must be a positive real scalar, not bool.")

    bandwidth_arr = np.asarray(bandwidth)
    if bandwidth_arr.ndim != 0:
        raise TypeError("bandwidth must be a scalar value.")

    try:
        bandwidth_value = float(bandwidth_arr)
    except (TypeError, ValueError) as exc:
        raise TypeError("bandwidth must be a positive real scalar.") from exc

    if not np.isfinite(bandwidth_value):
        raise ValueError("bandwidth must be finite.")

    if bandwidth_value <= 0:
        raise ValueError("bandwidth must be greater than zero.")

    return bandwidth_value


def _validate_kernel_inputs(
    distances: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, float]:
    """Validate common kernel inputs."""
    return _validate_distances(distances), _validate_bandwidth(bandwidth)


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute Gaussian kernel weights.

    Args:
        distances: Non-negative distances from a regression location.
        bandwidth: Positive bandwidth controlling the rate of weight decay.

    Returns:
        ndarray: Floating-point weights in the interval [0, 1]. Values may underflow
            to exactly zero for extremely large normalized distances.

    Notes:
        The kernel is defined as:

            w(d) = exp(-0.5 * (d / bandwidth) ** 2)
    """
    distances_arr, bandwidth_value = _validate_kernel_inputs(
        distances,
        bandwidth,
    )
    normalized_distances = distances_arr / bandwidth_value
    return np.exp(-0.5 * normalized_distances**2)


def bisquare_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute bi-square (quartic) kernel weights.

    Args:
        distances: Non-negative distances from a regression location.
        bandwidth: Positive bandwidth. Observations at or beyond the bandwidth receive
            zero weight.

    Returns:
        ndarray: Floating-point weights in the interval [0, 1].

    Notes:
        The kernel is defined as:

            w(d) = (1 - (d / bandwidth) ** 2) ** 2,  if d < bandwidth
            w(d) = 0,                                if d >= bandwidth
    """
    distances_arr, bandwidth_value = _validate_kernel_inputs(
        distances,
        bandwidth,
    )

    weights = np.zeros_like(distances_arr, dtype=float)
    mask = distances_arr < bandwidth_value
    normalized_distances = distances_arr[mask] / bandwidth_value
    weights[mask] = (1.0 - normalized_distances**2) ** 2
    return weights


def exponential_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute exponential kernel weights.

    The exponential kernel decreases more sharply near the origin than the
    Gaussian kernel, while retaining a heavier tail at large distances.

    Args:
        distances: Non-negative distances from a regression location.
        bandwidth: Positive bandwidth controlling the rate of weight decay.

    Returns:
        ndarray: Floating-point weights in the interval [0, 1]. Values may underflow
            to exactly zero for extremely large normalized distances.

    Notes:
        The kernel is defined as:

            w(d) = exp(-d / bandwidth)
    """
    distances_arr, bandwidth_value = _validate_kernel_inputs(
        distances,
        bandwidth,
    )
    return np.exp(-distances_arr / bandwidth_value)


def tricube_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute tri-cube kernel weights.

    Args:
        distances: Non-negative distances from a regression location.
        bandwidth: Positive bandwidth. Observations at or beyond the bandwidth receive
            zero weight.

    Returns:
        ndarray: Floating-point weights in the interval [0, 1].

    Notes:
        The kernel is defined as:

            w(d) = (1 - (d / bandwidth) ** 3) ** 3,  if d < bandwidth
            w(d) = 0,                                if d >= bandwidth
    """
    distances_arr, bandwidth_value = _validate_kernel_inputs(
        distances,
        bandwidth,
    )

    weights = np.zeros_like(distances_arr, dtype=float)
    mask = distances_arr < bandwidth_value
    normalized_distances = distances_arr[mask] / bandwidth_value
    weights[mask] = (1.0 - normalized_distances**3) ** 3
    return weights


def boxcar_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute boxcar (uniform) kernel weights.

    Args:
        distances: Non-negative distances from a regression location.
        bandwidth: Positive bandwidth. Observations within or exactly on the bandwidth
            boundary receive unit weight.

    Returns:
        ndarray: Floating-point weights containing only 0 and 1.

    Notes:
        The kernel is defined as:

            w(d) = 1,  if d <= bandwidth
            w(d) = 0,  if d > bandwidth
    """
    distances_arr, bandwidth_value = _validate_kernel_inputs(
        distances,
        bandwidth,
    )
    return (distances_arr <= bandwidth_value).astype(float)


# Built-in kernel registry.
KERNEL_FUNCTIONS: dict[str, KernelCallable] = {
    "gaussian": gaussian_kernel,
    "bisquare": bisquare_kernel,
    "exponential": exponential_kernel,
    "tricube": tricube_kernel,
    "boxcar": boxcar_kernel,
}


def get_kernel_function(kernel: KernelLike) -> KernelCallable:
    """Return a built-in kernel by name or validate a custom callable.

    Args:
        kernel: Built-in kernel name or a callable with the signature
            ``kernel(distances, bandwidth) -> weights``.

    Returns:
        callable: Kernel function.

    Raises:
        TypeError: If ``kernel`` is neither a string nor a callable.
        ValueError: If a string kernel name is unknown or empty.
    """
    if callable(kernel):
        return kernel

    if not isinstance(kernel, str):
        raise TypeError(
            "kernel must be a string name or callable; " f"got {type(kernel).__name__}."
        )

    kernel_name = kernel.strip().lower()
    if not kernel_name:
        raise ValueError("kernel name cannot be empty.")

    try:
        return KERNEL_FUNCTIONS[kernel_name]
    except KeyError as exc:
        available = ", ".join(sorted(KERNEL_FUNCTIONS))
        raise ValueError(
            f"Unknown kernel: {kernel!r}. Available kernels: {available}."
        ) from exc
