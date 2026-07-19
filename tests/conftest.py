"""Shared pytest fixtures for the pyGWRx test suite.

Provides small synthetic datasets so tests run fast and require no external
data files. A ``src/`` fallback is added to ``sys.path`` so the suite works
whether or not the package has been installed in editable mode.
"""

import os
import sys

import numpy as np
import pytest

# Allow running the tests directly from a checkout without `pip install -e .`
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session")
def synthetic():
    """A small, well-conditioned synthetic spatial regression dataset.

    Returns a dict with ``X`` (n, 3), ``y`` (n,), ``coords`` (n, 2),
    ``times`` (n,), and ``beta`` (the true global coefficients).
    """
    rng = np.random.default_rng(42)
    n = 60
    coords = rng.random((n, 2)) * 10.0
    X = rng.random((n, 3))
    beta = np.array([1.0, -2.0, 0.5])
    # add a mild spatial trend so local coefficients are meaningful
    y = X @ beta + coords[:, 0] * 0.2 + rng.normal(0.0, 0.1, n)
    times = np.sort(rng.random(n) * 5.0)
    return {"X": X, "y": y, "coords": coords, "times": times, "beta": beta, "n": n}
