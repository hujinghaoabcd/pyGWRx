from pathlib import Path

GWR = Path("src/pygwrx/models/gwr.py")
DOC = Path("docs/models/gwr.md")
TEST = Path("tests/test_gwr_freeze_contract.py")

gwr = GWR.read_text(encoding="utf-8")
old_fields = '''    coef_t_values: Optional[np.ndarray] = None\n    intercept_t_values: Optional[np.ndarray] = None\n\n    def to_frame(self) -> pd.DataFrame:\n'''
new_fields = '''    coef_t_values: Optional[np.ndarray] = None\n    intercept_t_values: Optional[np.ndarray] = None\n    local_rank: Optional[np.ndarray] = None\n    local_condition_number: Optional[np.ndarray] = None\n    rank_deficient: Optional[np.ndarray] = None\n\n    def to_frame(self) -> pd.DataFrame:\n'''
assert gwr.count(old_fields) == 1
gwr = gwr.replace(old_fields, new_fields, 1)
old_frame = '''        if self.intercept_t_values is not None:\n            data["intercept_t"] = self.intercept_t_values\n\n        for index, name in enumerate(self.feature_names):\n'''
new_frame = '''        if self.intercept_t_values is not None:\n            data["intercept_t"] = self.intercept_t_values\n        if self.local_rank is not None:\n            data["local_rank"] = self.local_rank\n        if self.local_condition_number is not None:\n            data["local_condition_number"] = self.local_condition_number\n        if self.rank_deficient is not None:\n            data["rank_deficient"] = self.rank_deficient\n\n        for index, name in enumerate(self.feature_names):\n'''
assert gwr.count(old_frame) == 1
gwr = gwr.replace(old_frame, new_frame, 1)
old_reset = '''        self.S_matrix_ = None\n        self.bandwidth_search_ = None\n\n    def _resolve_bandwidth(\n'''
new_reset = '''        self.S_matrix_ = None\n        self.bandwidth_search_ = None\n        self.n_samples_ = None\n        self.n_features_in_ = None\n        self.feature_names_in_ = None\n\n    def _resolve_bandwidth(\n'''
assert gwr.count(old_reset) == 1
gwr = gwr.replace(old_reset, new_reset, 1)
old_fit_start = '''        ``compute_hat_matrix_flag`` is retained as a compatibility alias for older\n        PyGWRx code. New code should use ``compute_hat_matrix``.\n        """\n        if compute_hat_matrix_flag is not None:\n'''
new_fit_start = '''        ``compute_hat_matrix_flag`` is retained as a compatibility alias for older\n        PyGWRx code. New code should use ``compute_hat_matrix``.\n        """\n        self._reset_fit_state()\n        if compute_hat_matrix_flag is not None:\n'''
assert gwr.count(old_fit_start) == 1
gwr = gwr.replace(old_fit_start, new_fit_start, 1)
old_late_reset = '''        if not isinstance(self.sigma2_v1, (bool, np.bool_)):\n            raise TypeError("sigma2_v1 must be boolean.")\n        self._reset_fit_state()\n\n        try:\n'''
new_late_reset = '''        if not isinstance(self.sigma2_v1, (bool, np.bool_)):\n            raise TypeError("sigma2_v1 must be boolean.")\n\n        try:\n'''
assert gwr.count(old_late_reset) == 1
gwr = gwr.replace(old_late_reset, new_late_reset, 1)
old_pred = '''            inverse_xtx_xtw = solve.inverse_normal @ (X_design.T * weights)\n            full_params[index] = solve.beta\n            local_rank[index] = solve.rank\n            local_condition_number[index] = solve.condition_number\n            if covariance_factors is not None:\n                if solve.rank < X_design.shape[1]:\n                    covariance_factors[index] = np.nan\n                else:\n                    covariance_factors[index] = np.sum(\n                        inverse_xtx_xtw**2,\n                        axis=1,\n                    )\n'''
new_pred = '''            full_params[index] = solve.beta\n            local_rank[index] = solve.rank\n            local_condition_number[index] = solve.condition_number\n            if covariance_factors is not None:\n                if solve.rank < X_design.shape[1]:\n                    covariance_factors[index] = np.nan\n                else:\n                    inverse_xtx_xtw = solve.inverse_normal @ (X_design.T * weights)\n                    covariance_factors[index] = np.sum(\n                        inverse_xtx_xtw**2,\n                        axis=1,\n                    )\n'''
assert gwr.count(old_pred) == 1
gwr = gwr.replace(old_pred, new_pred, 1)
old_result = '''            coef_t_values=coef_t,\n            intercept_t_values=intercept_t,\n        )\n'''
new_result = '''            coef_t_values=coef_t,\n            intercept_t_values=intercept_t,\n            local_rank=np.asarray(params["local_rank"], dtype=int),\n            local_condition_number=np.asarray(\n                params["local_condition_number"], dtype=float\n            ),\n            rank_deficient=np.asarray(params["rank_deficient"], dtype=bool),\n        )\n'''
assert gwr.count(old_result) == 1
gwr = gwr.replace(old_result, new_result, 1)
GWR.write_text(gwr, encoding="utf-8")

doc = DOC.read_text(encoding="utf-8")
old = '| `predict_result()` | Predictions, local slopes, intercepts, coordinates and optional standard errors/t statistics | Auditable prediction and coefficient inspection. Rank-deficient target recalibrations keep predictions but expose `NaN` coefficient inference. |'
new = '| `predict_result()` | Predictions, local slopes, intercepts, coordinates, rank/condition diagnostics and optional standard errors/t statistics | Auditable prediction and coefficient inspection. Rank-deficient target recalibrations keep predictions, set coefficient inference to `NaN`, and expose the numerical-rank flags directly. |'
assert doc.count(old) == 1
doc = doc.replace(old, new, 1)
DOC.write_text(doc, encoding="utf-8")

test = TEST.read_text(encoding="utf-8")
test += '''\n\ndef test_failed_refit_clears_previous_fitted_state():\n    X, y, coords = _data()\n    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)\n    assert model.is_fitted_\n\n    with pytest.raises(TypeError, match="compute_hat_matrix must be boolean"):\n        model.fit(X, y, coords, compute_hat_matrix="yes")\n\n    assert not model.is_fitted_\n    assert model.n_samples_ is None\n    assert model.n_features_in_ is None\n    assert model.feature_names_in_ is None\n    assert model.X_train_ is None\n    assert model.y_train_ is None\n    assert model.coords_train_ is None\n    assert model.coef_ is None\n    assert model.intercept_ is None\n    assert model.bandwidth_ is None\n    assert model.bandwidth_search_ is None\n\n\ndef test_failed_parameter_validation_refit_also_clears_previous_state():\n    X, y, coords = _data()\n    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)\n    model.bandwidth = -1.0\n\n    with pytest.raises(ValueError, match="numeric bandwidth"):\n        model.fit(X, y, coords)\n\n    assert not model.is_fitted_\n    assert model.n_samples_ is None\n    assert model.coef_ is None\n    assert model.bandwidth_ is None\n\n\ndef test_prediction_result_exposes_rank_diagnostics_even_without_inference():\n    X, y, coords = _data()\n    model = GWR(kernel="gaussian", bandwidth=0.8).fit(\n        X, y, coords, compute_inference=False\n    )\n    result = model.predict_result(X[:4], coords[:4])\n\n    assert result.local_rank is not None\n    assert result.local_condition_number is not None\n    assert result.rank_deficient is not None\n    assert result.local_rank.shape == (4,)\n    assert result.local_condition_number.shape == (4,)\n    assert result.rank_deficient.shape == (4,)\n    assert result.coef_standard_errors is None\n    frame = result.to_frame()\n    assert "local_rank" in frame\n    assert "local_condition_number" in frame\n    assert "rank_deficient" in frame\n'''
TEST.write_text(test, encoding="utf-8")
