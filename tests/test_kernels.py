"""Basic mathematical properties of the spatial kernel functions."""

import numpy as np
import pytest

from pygwrx import bisquare_kernel, exponential_kernel, gaussian_kernel

KERNELS = [gaussian_kernel, bisquare_kernel, exponential_kernel]


@pytest.mark.parametrize("kernel", KERNELS)
def test_weights_in_unit_interval(kernel):
    d = np.linspace(0.0, 5.0, 50)
    w = np.asarray(kernel(d, bandwidth=2.0))
    assert w.shape == d.shape
    assert np.all(w >= -1e-12)
    assert np.all(w <= 1.0 + 1e-9)


@pytest.mark.parametrize("kernel", KERNELS)
def test_zero_distance_has_max_weight(kernel):
    d = np.array([0.0, 0.5, 1.0, 2.0])
    w = np.asarray(kernel(d, bandwidth=2.0))
    # weight at distance 0 should be the largest
    assert w[0] == pytest.approx(np.max(w))


@pytest.mark.parametrize("kernel", KERNELS)
def test_monotonic_non_increasing(kernel):
    d = np.linspace(0.0, 3.0, 40)
    w = np.asarray(kernel(d, bandwidth=1.5))
    assert np.all(np.diff(w) <= 1e-9)
