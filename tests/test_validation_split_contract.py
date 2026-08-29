# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Architecture contracts for the B2 validation-module split.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pygwrx.core as core
from pygwrx.core import distance, utils, validation


def test_validation_module_is_canonical_and_legacy_utils_reexports() -> None:
    """The validation module owns validation while old imports remain aliases."""
    assert validation.validate_coords.__module__ == "pygwrx.core.validation"
    assert validation.validate_data.__module__ == "pygwrx.core.validation"

    assert utils.validate_coords is validation.validate_coords
    assert utils.validate_data is validation.validate_data
    assert core.validate_coords is validation.validate_coords
    assert core.validate_data is validation.validate_data


def test_validation_split_keeps_later_phase_responsibilities_separate() -> None:
    """Validation remains isolated as later Phase-B owners are introduced."""
    assert not hasattr(validation, "compute_distance_matrix")
    assert not hasattr(validation, "DistanceCache")
    assert not hasattr(validation, "chunked_computation")
    assert not hasattr(validation, "add_intercept")

    assert utils.compute_distance_matrix is distance.compute_distance_matrix
    assert utils.DistanceCache is distance.DistanceCache
    assert utils.chunked_computation is distance.chunked_computation
    assert utils.add_intercept.__module__ == "pygwrx.core.utils"


def test_legacy_and_canonical_validation_results_match() -> None:
    """Representative validation behavior is unchanged through the compatibility path."""
    coords = pd.DataFrame({"x": [0, 1], "y": [2, 3]})
    X = pd.DataFrame({"feature": [1, 2]})
    y = pd.Series([3, 4])

    np.testing.assert_array_equal(
        utils.validate_coords(coords), validation.validate_coords(coords)
    )
    legacy_X, legacy_y = utils.validate_data(X, y)
    canonical_X, canonical_y = validation.validate_data(X, y)
    np.testing.assert_array_equal(legacy_X, canonical_X)
    np.testing.assert_array_equal(legacy_y, canonical_y)
