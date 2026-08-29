# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Parametric-bootstrap inference for coefficient non-stationarity in GWR.

The implementation follows the multiple-linear-regression (MLR) null model in
Harris et al. (2017) and the maintained ``GWmodel::gwr.bootstrap`` source.  It
reports the coefficient-wise modified statistic and complementary localised
pseudo-t statistics.  Spatial-error, spatial-moving-average, and spatial-lag
null models remain outside the supported Python API until direct numerical
validation is available.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx.core._summary import format_summary
from pygwrx.core.utils import add_intercept, validate_coords
from pygwrx.models.gwr import GWR


class BootstrapGWR:
    r"""Test GWR coefficient non-stationarity by parametric bootstrap.

    For coefficient :math:`j`, the modified statistic is the sample standard
    deviation across locations of the fitted GWR pseudo-t values,

    .. math::

        T_j = \operatorname{sd}_i\left(\hat\beta_j(s_i) / \widehat{se}_j(s_i)\right).

    The localised statistic compares each local coefficient with the matching
    coefficient from the global null model,

    .. math::

        t_{ij}^{\mathrm{loc}} =
        \frac{\hat\beta_j(s_i)-\hat\beta_j^{\mathrm{OLS}}}
             {\widehat{se}_j(s_i)}.

    Bootstrap responses are generated parametrically under the OLS null as
    ``X @ beta_ols + Normal(0, sigma_ols)``.  By default, an automatically
    selected GWR bandwidth is selected again in every bootstrap replicate, as
    in ``GWmodel``.  Set ``reselect_bandwidth=False`` to condition on the
    observed selected bandwidth.

    Args:
        bandwidth: Numeric GWR bandwidth, automatic criterion (``"cv"``,
            ``"aic"``, ``"aicc"``, or ``"bic"``), or ``None``.
        adaptive: Interpret a numeric or selected bandwidth as a neighbour count.
        kernel: GWR spatial kernel.
        bandwidth_method: Criterion used when ``bandwidth=None``.
        bandwidth_range: Optional search interval for automatic bandwidth selection.
        optimization_method: Bandwidth search method forwarded to :class:`GWR`.
        fit_intercept: Include an intercept in both GWR and OLS null models.
        distance_metric: Distance metric forwarded to :class:`GWR`.
        n_bootstrap: Number of parametric bootstrap replicates.
        reselect_bandwidth: Re-select an automatic bandwidth in each replicate.
        pvalue_method: ``"plus_one"`` uses the finite-sample correction
            ``(1 + exceedances) / (R + 1)``. ``"gwmodel"`` reproduces the
            maintained R helper ``exceedances / (R + 1)``.
        localized_tail: ``"two-sided"`` compares absolute pseudo-t statistics;
            ``"right"`` reproduces the one-sided comparison in GWmodel.
        store_local_bootstrap: Store the full ``R x n x p`` local statistic array.
            Local p-values are available regardless of this option.
        random_state: Seed or NumPy generator for reproducible simulation.
        verbose: Print bootstrap progress.

    Attributes:
        modified_statistics_: Observed coefficient-wise modified statistics.
        modified_p_values_: Bootstrap right-tail p-values for the modified test.
        localized_statistics_: Observed ``n x p`` localised pseudo-t statistics.
        localized_p_values_: Observation- and coefficient-specific bootstrap p-values.
        coefficients_gwr_: Full local parameter matrix including the intercept.
        coefficients_global_: OLS null-model parameter vector.
        bandwidth_: Bandwidth selected for the observed GWR fit.

    References:
        Harris, P., Brunsdon, C., Lu, B., Nakaya, T., & Charlton, M. (2017).
        Introducing bootstrap methods to investigate coefficient
        non-stationarity in spatial regression models. *Spatial Statistics*,
        21, 241-261. https://doi.org/10.1016/j.spasta.2017.07.006
    """

    _AUTO_BANDWIDTHS = {"cv", "aic", "aicc", "bic"}

    def __init__(
        self,
        bandwidth: Union[float, int, str, None] = "aicc",
        adaptive: bool = False,
        kernel: str = "bisquare",
        bandwidth_method: str = "aicc",
        bandwidth_range: Optional[Tuple[float, float]] = None,
        optimization_method: str = "golden_section",
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        n_bootstrap: int = 99,
        reselect_bandwidth: bool = True,
        pvalue_method: str = "plus_one",
        localized_tail: str = "two-sided",
        store_local_bootstrap: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        verbose: bool = False,
    ) -> None:
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.kernel = kernel
        self.bandwidth_method = bandwidth_method
        self.bandwidth_range = bandwidth_range
        self.optimization_method = optimization_method
        self.fit_intercept = fit_intercept
        self.distance_metric = distance_metric
        self.n_bootstrap = n_bootstrap
        self.reselect_bandwidth = reselect_bandwidth
        self.pvalue_method = pvalue_method
        self.localized_tail = localized_tail
        self.store_local_bootstrap = store_local_bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        self._is_fitted = False
        self.n_samples_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.parameter_names_: Optional[Tuple[str, ...]] = None
        self.bandwidth_: Optional[Union[int, float]] = None
        self.gwr_model_: Optional[GWR] = None
        self.coefficients_gwr_: Optional[np.ndarray] = None
        self.local_standard_errors_: Optional[np.ndarray] = None
        self.local_t_values_: Optional[np.ndarray] = None
        self.fitted_values_gwr_: Optional[np.ndarray] = None
        self.residuals_gwr_: Optional[np.ndarray] = None
        self.coefficients_global_: Optional[np.ndarray] = None
        self.fitted_values_global_: Optional[np.ndarray] = None
        self.residuals_global_: Optional[np.ndarray] = None
        self.sigma_global_: Optional[float] = None
        self.modified_statistics_: Optional[np.ndarray] = None
        self.modified_critical_values_: Optional[np.ndarray] = None
        self.modified_p_values_: Optional[np.ndarray] = None
        self.bootstrap_modified_statistics_: Optional[np.ndarray] = None
        self.localized_statistics_: Optional[np.ndarray] = None
        self.localized_p_values_: Optional[np.ndarray] = None
        self.bootstrap_localized_statistics_: Optional[np.ndarray] = None
        self.localized_lower_critical_: Optional[np.ndarray] = None
        self.localized_upper_critical_: Optional[np.ndarray] = None
        self.test_statistic_: Optional[np.ndarray] = None
        self.bootstrap_statistics_: Optional[np.ndarray] = None
        self.p_value_: Optional[np.ndarray] = None
        self.rss_gwr_: Optional[float] = None
        self.rss_global_: Optional[float] = None
        self.trace_S_: Optional[float] = None
        self.bootstrap_bandwidths_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.coords_train_: Optional[np.ndarray] = None

    def _validate_parameters(self) -> None:
        if not isinstance(self.n_bootstrap, (int, np.integer)):
            raise TypeError("n_bootstrap must be an integer.")
        if int(self.n_bootstrap) < 1:
            raise ValueError("n_bootstrap must be at least 1.")
        for name, value in (
            ("adaptive", self.adaptive),
            ("fit_intercept", self.fit_intercept),
            ("reselect_bandwidth", self.reselect_bandwidth),
            ("store_local_bootstrap", self.store_local_bootstrap),
            ("verbose", self.verbose),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")

        pvalue_method = str(self.pvalue_method).strip().lower()
        if pvalue_method not in {"plus_one", "gwmodel"}:
            raise ValueError("pvalue_method must be 'plus_one' or 'gwmodel'.")
        localized_tail = str(self.localized_tail).strip().lower()
        if localized_tail not in {"two-sided", "right"}:
            raise ValueError("localized_tail must be 'two-sided' or 'right'.")

    @staticmethod
    def _validate_data(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        names = None
        if isinstance(X, pd.DataFrame):
            names = np.asarray([str(column) for column in X.columns], dtype=object)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if X_arr.shape[1] < 1:
            raise ValueError("X must contain at least one predictor.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values.")

        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains NaN or infinite values.")

        coords_arr = validate_coords(coords)
        if coords_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                "X and coords must contain the same number of observations."
            )
        return X_arr, y_arr, coords_arr, names

    def _make_gwr(self, bandwidth: Union[float, int, str, None]) -> GWR:
        return GWR(
            kernel=self.kernel,
            bandwidth=bandwidth,
            bandwidth_method=self.bandwidth_method,
            adaptive=self.adaptive,
            bandwidth_range=self.bandwidth_range,
            optimization_method=self.optimization_method,
            fit_intercept=self.fit_intercept,
            distance_metric=self.distance_metric,
            verbose=False,
        )

    @staticmethod
    def _full_local_parameters(model: GWR) -> Tuple[np.ndarray, np.ndarray]:
        if model.rank_deficient_ is not None and np.any(model.rank_deficient_):
            count = int(np.count_nonzero(model.rank_deficient_))
            raise np.linalg.LinAlgError(
                f"The fitted GWR contains {count} rank deficient local weighted "
                "design(s); BootstrapGWR requires identifiable local coefficient "
                "inference."
            )
        if model.coef_ is None or model.coef_se_ is None:
            raise RuntimeError("The fitted GWR model did not produce local inference.")
        if model.fit_intercept:
            if model.intercept_ is None or model.intercept_se_ is None:
                raise RuntimeError("The fitted GWR intercept inference is unavailable.")
            params = np.column_stack([model.intercept_, model.coef_])
            standard_errors = np.column_stack([model.intercept_se_, model.coef_se_])
        else:
            params = np.asarray(model.coef_, dtype=float)
            standard_errors = np.asarray(model.coef_se_, dtype=float)
        if not np.all(np.isfinite(params)):
            raise ValueError("The GWR fit produced non-finite local coefficients.")
        if not np.all(np.isfinite(standard_errors)) or np.any(standard_errors <= 0.0):
            raise ValueError(
                "The GWR fit produced non-positive or non-finite local standard errors."
            )
        return params, standard_errors

    @staticmethod
    def _fit_ols(
        X_design: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        rank = int(np.linalg.matrix_rank(X_design))
        if rank < X_design.shape[1]:
            raise np.linalg.LinAlgError(
                "The global OLS design matrix is rank deficient."
            )
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        fitted = X_design @ beta
        residuals = y - fitted
        rss = float(np.dot(residuals, residuals))
        residual_df = X_design.shape[0] - rank
        if residual_df <= 0:
            raise ValueError(
                "The OLS null model has no positive residual degrees of freedom."
            )
        sigma = float(np.sqrt(rss / residual_df))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                "The OLS null model has zero or invalid residual variance."
            )
        return beta, fitted, residuals, rss, sigma

    @staticmethod
    def _modified_statistic(
        local_parameters: np.ndarray, local_standard_errors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        local_t = local_parameters / local_standard_errors
        statistic = np.std(local_t, axis=0, ddof=1)
        return statistic, local_t

    @staticmethod
    def _localized_statistic(
        local_parameters: np.ndarray,
        global_parameters: np.ndarray,
        local_standard_errors: np.ndarray,
    ) -> np.ndarray:
        return (
            local_parameters - global_parameters[np.newaxis, :]
        ) / local_standard_errors

    def _bootstrap_p_values(self, exceedances: np.ndarray) -> np.ndarray:
        denominator = float(self.n_bootstrap + 1)
        if str(self.pvalue_method).strip().lower() == "plus_one":
            return (1.0 + exceedances.astype(float)) / denominator
        return exceedances.astype(float) / denominator

    def _bootstrap_bandwidth_spec(self) -> Union[float, int, str, None]:
        if self.reselect_bandwidth:
            return self.bandwidth
        if self.bandwidth_ is None:
            raise RuntimeError("The observed fitted bandwidth is unavailable.")
        return self.bandwidth_

    def _rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> "BootstrapGWR":
        """Fit the observed models and run the MLR-null parametric bootstrap."""
        self._reset_fit_state()
        self._validate_parameters()
        try:
            X_arr, y_arr, coords_arr, names = self._validate_data(X, y, coords)
            X_design = add_intercept(X_arr) if self.fit_intercept else X_arr.copy()
            parameter_names = (("intercept",) if self.fit_intercept else ()) + tuple(
                str(name)
                for name in (
                    names
                    if names is not None
                    else [f"x{i}" for i in range(X_arr.shape[1])]
                )
            )

            observed_gwr = self._make_gwr(self.bandwidth)
            observed_gwr.fit(
                X_arr,
                y_arr,
                coords_arr,
                compute_hat_matrix=False,
                compute_local_r2=False,
                compute_inference=True,
            )
            local_params, local_se = self._full_local_parameters(observed_gwr)
            global_beta, global_fitted, global_residuals, global_rss, global_sigma = (
                self._fit_ols(X_design, y_arr)
            )
            modified, local_t = self._modified_statistic(local_params, local_se)
            localized = self._localized_statistic(local_params, global_beta, local_se)

            n_parameters = X_design.shape[1]
            bootstrap_modified = np.empty((self.n_bootstrap, n_parameters), dtype=float)
            bootstrap_local = (
                np.empty((self.n_bootstrap, X_arr.shape[0], n_parameters), dtype=float)
                if self.store_local_bootstrap
                else None
            )
            modified_exceedances = np.zeros(n_parameters, dtype=np.int64)
            localized_exceedances = np.zeros(
                (X_arr.shape[0], n_parameters), dtype=np.int64
            )
            bootstrap_bandwidths = np.empty(self.n_bootstrap, dtype=float)
            rng = self._rng()
            bandwidth_spec = (
                self.bandwidth if self.reselect_bandwidth else observed_gwr.bandwidth_
            )
            tail = str(self.localized_tail).strip().lower()

            for index in range(self.n_bootstrap):
                y_boot = global_fitted + rng.normal(
                    loc=0.0, scale=global_sigma, size=X_arr.shape[0]
                )
                boot_gwr = self._make_gwr(bandwidth_spec)
                boot_gwr.fit(
                    X_arr,
                    y_boot,
                    coords_arr,
                    compute_hat_matrix=False,
                    compute_local_r2=False,
                    compute_inference=True,
                )
                boot_params, boot_se = self._full_local_parameters(boot_gwr)
                boot_global_beta, _, _, _, _ = self._fit_ols(X_design, y_boot)
                boot_modified, _ = self._modified_statistic(boot_params, boot_se)
                boot_localized = self._localized_statistic(
                    boot_params, boot_global_beta, boot_se
                )

                bootstrap_modified[index] = boot_modified
                if bootstrap_local is not None:
                    bootstrap_local[index] = boot_localized
                modified_exceedances += boot_modified >= modified
                if tail == "two-sided":
                    localized_exceedances += np.abs(boot_localized) >= np.abs(localized)
                else:
                    localized_exceedances += boot_localized > localized
                bootstrap_bandwidths[index] = float(boot_gwr.bandwidth_)

                if self.verbose:
                    step = max(1, self.n_bootstrap // 10)
                    if (index + 1) % step == 0 or index + 1 == self.n_bootstrap:
                        print(
                            f"BootstrapGWR: completed {index + 1}/{self.n_bootstrap} replicates."
                        )

            modified_p = self._bootstrap_p_values(modified_exceedances)
            localized_p = self._bootstrap_p_values(localized_exceedances)

            self.n_samples_ = X_arr.shape[0]
            self.n_features_in_ = X_arr.shape[1]
            self.feature_names_in_ = None if names is None else names.copy()
            self.parameter_names_ = parameter_names
            self.bandwidth_ = observed_gwr.bandwidth_
            self.gwr_model_ = observed_gwr
            self.coefficients_gwr_ = local_params
            self.local_standard_errors_ = local_se
            self.local_t_values_ = local_t
            self.fitted_values_gwr_ = np.asarray(
                observed_gwr.fitted_values_, dtype=float
            )
            self.residuals_gwr_ = np.asarray(observed_gwr.residuals_, dtype=float)
            self.coefficients_global_ = global_beta
            self.fitted_values_global_ = global_fitted
            self.residuals_global_ = global_residuals
            self.sigma_global_ = global_sigma
            self.modified_statistics_ = modified
            self.modified_critical_values_ = np.quantile(
                bootstrap_modified, 0.95, axis=0
            )
            self.modified_p_values_ = modified_p
            self.bootstrap_modified_statistics_ = bootstrap_modified
            self.localized_statistics_ = localized
            self.localized_p_values_ = localized_p
            self.bootstrap_localized_statistics_ = bootstrap_local
            if bootstrap_local is not None:
                self.localized_lower_critical_ = np.quantile(
                    bootstrap_local, 0.025, axis=0
                )
                self.localized_upper_critical_ = np.quantile(
                    bootstrap_local, 0.975, axis=0
                )
            self.bootstrap_bandwidths_ = bootstrap_bandwidths
            self.rss_gwr_ = float(np.dot(self.residuals_gwr_, self.residuals_gwr_))
            self.rss_global_ = global_rss
            self.trace_S_ = float(observed_gwr.diagnostics_["trace_S"])

            # Compatibility aliases from the historical pyGWRx API.  The valid
            # Harris/GWmodel statistic is coefficient-wise rather than scalar.
            self.test_statistic_ = self.modified_statistics_.copy()
            self.bootstrap_statistics_ = self.bootstrap_modified_statistics_.copy()
            self.p_value_ = self.modified_p_values_.copy()
            self.X_train_ = X_arr.copy()
            self.y_train_ = y_arr.copy()
            self.coords_train_ = coords_arr.copy()
            self._is_fitted = True
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def to_frame(self) -> pd.DataFrame:
        """Return local coefficients, inference, and bootstrap p-values."""
        self._check_is_fitted()
        if (
            self.coords_train_ is None
            or self.coefficients_gwr_ is None
            or self.local_standard_errors_ is None
            or self.localized_statistics_ is None
            or self.localized_p_values_ is None
            or self.parameter_names_ is None
        ):
            raise RuntimeError("Stored BootstrapGWR results are incomplete.")
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords_train_[:, 0],
            "coord_1": self.coords_train_[:, 1],
        }
        for index, name in enumerate(self.parameter_names_):
            data[f"coef_{name}"] = self.coefficients_gwr_[:, index]
            data[f"se_{name}"] = self.local_standard_errors_[:, index]
            data[f"localized_t_{name}"] = self.localized_statistics_[:, index]
            data[f"localized_p_{name}"] = self.localized_p_values_[:, index]
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Return a plain-text summary of the bootstrap test."""
        self._check_is_fitted()
        return format_summary(
            "Bootstrap GWR Summary",
            {
                "parameter_names": self.parameter_names_,
                "bandwidth": self.bandwidth_,
                "n_bootstrap": self.n_bootstrap,
                "null_model": "ols",
                "modified_statistics": self.modified_statistics_.copy(),
                "modified_critical_values_95": self.modified_critical_values_.copy(),
                "modified_p_values": self.modified_p_values_.copy(),
                "rss_gwr": self.rss_gwr_,
                "rss_global": self.rss_global_,
                "trace_S": self.trace_S_,
                "reselect_bandwidth": self.reselect_bandwidth,
                "pvalue_method": self.pvalue_method,
                "localized_tail": self.localized_tail,
            },
        )

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
