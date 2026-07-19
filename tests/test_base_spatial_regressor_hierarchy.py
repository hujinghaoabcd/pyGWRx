# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Regression tests for the consolidated spatial regressor hierarchy."""

from pygwrx.core import (
    BaseGWR,
    BaseMultiscaleRegressor,
    BaseSpatialRegressor,
    BaseSpatiotemporalRegressor,
)
from pygwrx.models.gw_lasso import GWLasso
from pygwrx.models.gwr import GWR
from pygwrx.models.mgwr import MGWR
from pygwrx.models.mixed_gwr import MixedGWR


def test_base_gwr_is_backward_compatible_identity_alias() -> None:
    assert BaseGWR is BaseSpatialRegressor


def test_gwr_family_uses_base_spatial_regressor() -> None:
    for estimator in (GWR, GWLasso, MixedGWR, MGWR):
        assert issubclass(estimator, BaseSpatialRegressor)


def test_specialized_regressor_bases_use_consolidated_parent() -> None:
    assert issubclass(BaseSpatiotemporalRegressor, BaseSpatialRegressor)
    assert issubclass(BaseMultiscaleRegressor, BaseSpatialRegressor)
