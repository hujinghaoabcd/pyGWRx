from __future__ import annotations

import re
from pathlib import Path

solver_path = Path("src/pygwrx/core/solver.py")
solver = solver_path.read_text(encoding="utf-8")

solver, count = re.subn(
    r"\n\ndef _normal_equations\(.*?\n\ndef weighted_least_squares\(",
    "\n\ndef weighted_least_squares(",
    solver,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit("Could not remove obsolete production normal-equation helpers")

old_comment = (
    "    # A dummy y is used only to reuse the identical weighted normal-system builder.\n"
    "    dummy_y = np.zeros(X_arr.shape[0], dtype=float)\n"
)
new_comment = (
    "    # The response is irrelevant for the smoother matrix. A zero vector lets us\n"
    "    # reuse weighted_least_squares() so the hat matrix uses exactly the same\n"
    "    # SVD-based inverse-normal operator as local coefficient estimation.\n"
    "    dummy_y = np.zeros(X_arr.shape[0], dtype=float)\n"
)
if old_comment not in solver:
    raise SystemExit("Could not find the compute_hat_matrix dummy-response comment")
solver = solver.replace(old_comment, new_comment, 1)

old_block = '''            system, _, XtW = _normal_equations(
                X_arr,
                dummy_y,
                weights,
                ridge=ridge_value,
            )

            # X_i @ inv(system) @ XtW, computed without explicitly inverting system.
            left = _solve_linear_system(system.T, X_arr[i])
            hat_row = left @ XtW
'''
new_block = '''            _, inverse_normal = weighted_least_squares(
                X_arr,
                dummy_y,
                weights,
                ridge=ridge_value,
            )
            XtW = X_arr.T * weights
            hat_row = X_arr[i] @ inverse_normal @ XtW
'''
if old_block not in solver:
    raise SystemExit("Could not find the old compute_hat_matrix normal-equation block")
solver = solver.replace(old_block, new_block, 1)

solver_path.write_text(solver, encoding="utf-8", newline="\n")

test_path = Path("tests/test_solver_unpenalized.py")
test = test_path.read_text(encoding="utf-8")
test = test.replace(
    "from pygwrx.core import weighted_least_squares\n",
    "from pygwrx.core import compute_hat_matrix, local_regression, weighted_least_squares\n"
    "from pygwrx.core.kernels import gaussian_kernel\n",
    1,
)
append = r'''


def test_hat_matrix_matches_same_svd_wls_operator():
    rng = np.random.default_rng(314159)
    X = np.column_stack([np.ones(9), rng.normal(size=(9, 2))])
    coords = np.column_stack([np.linspace(0.0, 4.0, 9), rng.normal(scale=0.2, size=9)])
    bandwidth = 1.7

    hat = compute_hat_matrix(X, coords, gaussian_kernel, bandwidth)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    dummy_y = np.zeros(X.shape[0], dtype=float)

    expected = np.empty_like(hat)
    for i, dists in enumerate(distances):
        weights = gaussian_kernel(dists, bandwidth)
        _, inverse_normal = weighted_least_squares(X, dummy_y, weights)
        expected[i] = X[i] @ inverse_normal @ (X.T * weights)

    np.testing.assert_allclose(hat, expected, rtol=1e-11, atol=1e-12)


def test_rank_deficient_hat_matrix_matches_minimum_norm_local_predictions():
    x = np.linspace(-2.0, 2.0, 11)
    X = np.column_stack([np.ones_like(x), x, 2.0 * x])
    y = 2.5 + 3.2 * x
    coords = np.column_stack([x, 0.15 * x**2])
    bandwidth = 1.6

    hat = compute_hat_matrix(X, coords, gaussian_kernel, bandwidth)
    local_beta = local_regression(
        X,
        y,
        coords,
        coords,
        gaussian_kernel,
        bandwidth,
    )
    fitted_from_beta = np.einsum("ij,ij->i", X, local_beta)

    assert np.all(np.isfinite(hat))
    np.testing.assert_allclose(hat @ y, fitted_from_beta, rtol=1e-10, atol=1e-11)


def test_ridge_hat_matrix_matches_explicit_ridge_local_predictions():
    rng = np.random.default_rng(2718)
    X = np.column_stack([np.ones(10), rng.normal(size=(10, 2))])
    y = rng.normal(size=10)
    coords = np.column_stack([np.linspace(0.0, 3.0, 10), rng.normal(scale=0.1, size=10)])
    bandwidth = 1.2
    ridge = 0.15

    hat = compute_hat_matrix(
        X,
        coords,
        gaussian_kernel,
        bandwidth,
        ridge=ridge,
    )
    local_beta = local_regression(
        X,
        y,
        coords,
        coords,
        gaussian_kernel,
        bandwidth,
        ridge=ridge,
    )
    fitted_from_beta = np.einsum("ij,ij->i", X, local_beta)

    np.testing.assert_allclose(hat @ y, fitted_from_beta, rtol=1e-10, atol=1e-11)
'''
if "test_hat_matrix_matches_same_svd_wls_operator" not in test:
    test += append

test_path.write_text(test, encoding="utf-8", newline="\n")

legacy_test_path = Path("tests/test_solver_legacy_reference.py")
legacy_test = legacy_test_path.read_text(encoding="utf-8")
legacy_append = r'''


def test_production_solver_no_longer_contains_normal_equation_helpers():
    import pygwrx.core.solver as solver_module

    source = inspect.getsource(solver_module)
    assert "def _normal_equations(" not in source
    assert "def _solve_linear_system(" not in source
    assert "_legacy_solver" not in source
'''
if "test_production_solver_no_longer_contains_normal_equation_helpers" not in legacy_test:
    legacy_test += legacy_append
legacy_test_path.write_text(legacy_test, encoding="utf-8", newline="\n")
