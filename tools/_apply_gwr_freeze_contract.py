from pathlib import Path

BASE = Path("src/pygwrx/core/base.py")
GWR = Path("src/pygwrx/models/gwr.py")
DOC = Path("docs/models/gwr.md")
TEST = Path("tests/test_gwr_freeze_contract.py")

base = BASE.read_text(encoding="utf-8")
old = '''            if lower <= 0 or upper <= 0 or lower > upper:\n                raise ValueError("bandwidth_range must satisfy 0 < lower <= upper.")\n            if adaptive and (not lower.is_integer() or not upper.is_integer()):\n'''
new = '''            if adaptive:\n                if lower <= 0 or upper <= 0 or lower > upper:\n                    raise ValueError(\n                        "adaptive bandwidth_range must satisfy 0 < lower <= upper."\n                    )\n            elif lower <= 0 or upper <= 0 or lower >= upper:\n                raise ValueError(\n                    "fixed bandwidth_range must satisfy 0 < lower < upper."\n                )\n            if adaptive and (not lower.is_integer() or not upper.is_integer()):\n'''
assert base.count(old) == 1
BASE.write_text(base.replace(old, new), encoding="utf-8")

gwr = GWR.read_text(encoding="utf-8")
assert gwr.count("        compute_hat_matrix: bool = True,\n") == 1
gwr = gwr.replace("        compute_hat_matrix: bool = True,\n", "        compute_hat_matrix: bool = False,\n", 1)
old_doc = '''        numeric-bandwidth fit does not also retain an ``n x n`` distance matrix.\n        Automatic bandwidth selection has its own distance-matrix policy.\n'''
new_doc = '''        numeric-bandwidth fit does not also retain an ``n x n`` distance matrix.\n        Automatic bandwidth selection uses the same bounded distance backend.\n'''
assert gwr.count(old_doc) == 1
gwr = gwr.replace(old_doc, new_doc, 1)
old_sigma = "        if np.isfinite(self.sigma2_) and self.sigma2_ >= 0.0:\n"
new_sigma = "        if np.isfinite(self.sigma2_) and self.sigma2_ > np.finfo(float).eps:\n"
assert gwr.count(old_sigma) == 1
gwr = gwr.replace(old_sigma, new_sigma, 1)
old_summary = '''        global_beta = np.linalg.lstsq(X_global, self.y_train_, rcond=None)[0]\n        global_fitted = X_global @ global_beta\n        global_residuals = self.y_train_ - global_fitted\n        global_rss = float(np.dot(global_residuals, global_residuals))\n        n, p = X_global.shape\n        global_df = max(n - p, 1)\n        global_sigma2 = global_rss / global_df\n        covariance = global_sigma2 * np.linalg.pinv(X_global.T @ X_global)\n        global_se = np.sqrt(np.maximum(np.diag(covariance), 0.0))\n'''
new_summary = '''        n, p = X_global.shape\n        global_solve = _weighted_least_squares_details(\n            X_global,\n            self.y_train_,\n            np.ones(n, dtype=float),\n        )\n        global_beta = global_solve.beta\n        global_fitted = X_global @ global_beta\n        global_residuals = self.y_train_ - global_fitted\n        global_rss = float(np.dot(global_residuals, global_residuals))\n        global_df = max(n - global_solve.rank, 1)\n        global_sigma2 = global_rss / global_df\n        if global_solve.rank < p:\n            global_se = np.full(p, np.nan, dtype=float)\n        else:\n            global_se = np.sqrt(\n                np.maximum(\n                    np.diag(global_solve.inverse_normal) * global_sigma2,\n                    0.0,\n                )\n            )\n'''
assert gwr.count(old_summary) == 1
gwr = gwr.replace(old_summary, new_summary, 1)
GWR.write_text(gwr, encoding="utf-8")

doc = DOC.read_text(encoding="utf-8")
assert doc.count("    compute_hat_matrix=True,\n") == 1
doc = doc.replace("    compute_hat_matrix=True,\n", "    compute_hat_matrix=False,\n", 1)
old_row = '| `compute_hat_matrix` | `True` | Stores the full `n × n` smoother matrix. Set to `False` for larger samples. The trace, `trace(S\'S)`, influence, AIC/AICc/BIC and effective-parameter diagnostics are still computed. |'
new_row = '| `compute_hat_matrix` | `False` | Does not store the full `n × n` smoother matrix by default. Set to `True` only when the matrix entries themselves are required. The trace, `trace(S\'S)`, influence, AIC/AICc/BIC and effective-parameter diagnostics are still computed. |'
assert doc.count(old_row) == 1
doc = doc.replace(old_row, new_row, 1)
DOC.write_text(doc, encoding="utf-8")

TEST.write_text('''# SPDX-FileCopyrightText: 2026 Jinghao Hu\n# SPDX-License-Identifier: MIT\n\n"""Freeze-contract regression tests for standard GWR."""\n\nimport numpy as np\nimport pytest\n\nfrom pygwrx import GWR\n\n\ndef _data(n_samples: int = 36):\n    rng = np.random.default_rng(20260829)\n    coords = rng.uniform(0.0, 1.0, size=(n_samples, 2))\n    X = rng.normal(size=(n_samples, 2))\n    y = 1.25 + 0.9 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(0.0, 0.05, n_samples)\n    return X, y, coords\n\n\ndef test_gwr_default_does_not_store_full_hat_matrix():\n    X, y, coords = _data()\n    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)\n\n    assert model.hat_matrix_ is None\n    assert model.S_matrix_ is None\n    assert model.diagnostics_ is not None\n    assert np.isfinite(model.diagnostics_["trace_S"])\n    assert np.isfinite(model.diagnostics_["trace_StS"])\n    assert model.influence_ is not None\n\n\ndef test_explicit_hat_matrix_storage_does_not_change_gwr_numerics():\n    X, y, coords = _data()\n    default = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)\n    stored = GWR(kernel="gaussian", bandwidth=0.8).fit(\n        X, y, coords, compute_hat_matrix=True\n    )\n\n    assert stored.hat_matrix_ is not None\n    np.testing.assert_allclose(default.coef_, stored.coef_, rtol=0.0, atol=0.0)\n    np.testing.assert_allclose(default.intercept_, stored.intercept_, rtol=0.0, atol=0.0)\n    np.testing.assert_allclose(default.fitted_values_, stored.fitted_values_, rtol=0.0, atol=0.0)\n    assert default.diagnostics_["trace_S"] == stored.diagnostics_["trace_S"]\n    assert default.diagnostics_["trace_StS"] == stored.diagnostics_["trace_StS"]\n\n\ndef test_fixed_equal_bandwidth_range_is_rejected_early():\n    with pytest.raises(ValueError, match=r"fixed bandwidth_range.*lower < upper"):\n        GWR(bandwidth="cv", adaptive=False, bandwidth_range=(1.0, 1.0))\n\n\ndef test_adaptive_equal_bandwidth_range_remains_valid_single_candidate():\n    X, y, coords = _data(24)\n    model = GWR(\n        kernel="gaussian",\n        bandwidth="cv",\n        adaptive=True,\n        bandwidth_range=(8, 8),\n    ).fit(X, y, coords, compute_local_r2=False)\n\n    assert model.bandwidth_ == 8\n    assert model.bandwidth_search_ is not None\n    assert model.bandwidth_search_["search_range"] == (8, 8)\n\n\ndef test_near_perfect_fit_leaves_undefined_residual_diagnostics_as_nan():\n    x = np.linspace(-1.0, 1.0, 30)\n    X = x[:, None]\n    y = 2.0 + 3.0 * x\n    coords = np.column_stack([x, np.zeros_like(x)])\n\n    model = GWR(kernel="gaussian", bandwidth=100.0).fit(\n        X, y, coords, compute_local_r2=False\n    )\n\n    assert model.sigma2_ is not None\n    assert model.sigma2_ <= np.finfo(float).eps\n    assert np.all(np.isnan(model.standardized_residuals_))\n    assert np.all(np.isnan(model.cooks_distance_))\n\n\ndef test_summary_uses_shared_rank_aware_solver_not_normal_equation_pinv(monkeypatch):\n    X, y, coords = _data()\n    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)\n\n    def forbidden(*args, **kwargs):\n        raise AssertionError("summary must not use a separate normal-equation pseudoinverse")\n\n    monkeypatch.setattr(np.linalg, "pinv", forbidden)\n    text = model.summary()\n    assert "Global OLS reference" in text\n    assert "GWR diagnostics" in text\n''', encoding="utf-8")
