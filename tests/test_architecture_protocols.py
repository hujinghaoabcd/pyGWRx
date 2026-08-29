# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""B1 architecture contracts for the private protocol spine."""

from __future__ import annotations

import pygwrx
import pygwrx.core as core
import pygwrx.core._protocols as protocols

EXPECTED_PROTOCOLS = {
    "FittedLifecycleProtocol",
    "RegressionSurfaceProtocol",
    "ParameterInferenceProtocol",
    "TemporalViewProtocol",
    "MultiscaleViewProtocol",
    "WeightProviderProtocol",
    "StoredWeightComponentsProtocol",
}


def test_b1_declares_only_the_frozen_protocol_capabilities():
    """The private module must expose the seven frozen structural capabilities."""
    declared = {
        name
        for name, value in vars(protocols).items()
        if name != "Protocol"
        and name.endswith("Protocol")
        and getattr(value, "_is_protocol", False)
    }
    assert declared == EXPECTED_PROTOCOLS
    assert protocols.__all__ == ()


def test_b1_protocols_are_not_public_package_exports():
    """B1 protocols stay private until a later API decision explicitly promotes them."""
    for name in EXPECTED_PROTOCOLS:
        assert name not in core.__all__
        assert not hasattr(core, name)
        assert not hasattr(pygwrx, name)
