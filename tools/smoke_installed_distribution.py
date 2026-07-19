# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exercise an installed pyGWRx distribution outside the source tree."""

from __future__ import annotations

import importlib.resources
import importlib.util

import numpy as np

import pygwrx
from pygwrx import MGTWR
from pygwrx.io import load_dataset


def main() -> None:
    """Verify package data, bundled data, and the self-contained MGTWR runtime."""
    if importlib.util.find_spec("mgtwr") is not None:
        raise RuntimeError("An external top-level 'mgtwr' package is installed.")

    typed_marker = importlib.resources.files("pygwrx").joinpath("py.typed")
    if not typed_marker.is_file():
        raise RuntimeError("The installed distribution is missing pygwrx/py.typed.")

    X_data, y_data, coords_data = load_dataset("columbus", return_type="arrays")
    if X_data.shape[0] != y_data.shape[0] or coords_data.shape[0] != y_data.shape[0]:
        raise RuntimeError("The bundled Columbus dataset has inconsistent rows.")

    rng = np.random.default_rng(9)
    n_samples = 16
    coords = np.column_stack(
        [np.linspace(0.0, 3.0, n_samples), np.sin(np.linspace(0.0, 2.0, n_samples))]
    )
    times = np.linspace(0.0, 2.0, n_samples)
    X = rng.normal(size=(n_samples, 1))
    y = 1.2 + 0.8 * X[:, 0] + 0.05 * rng.normal(size=n_samples)

    model = MGTWR(
        bandwidths=[10.0, 10.0],
        taus=[0.5, 0.5],
        adaptive=False,
        kernel="gaussian",
        calculate_inference=False,
        max_iter=20,
    ).fit(X, y, coords, times)
    if model.params_ is None or model.params_.shape != (n_samples, 2):
        raise RuntimeError("Installed MGTWR did not produce the expected parameters.")
    if model.fitted_values_ is None or not np.all(np.isfinite(model.fitted_values_)):
        raise RuntimeError("Installed MGTWR produced invalid fitted values.")

    print(
        f"pyGWRx {pygwrx.__version__}: distribution smoke test passed "
        f"(MGTWR R2={model.r2_:.6f})."
    )


if __name__ == "__main__":
    main()
