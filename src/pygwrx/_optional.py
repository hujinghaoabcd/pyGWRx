# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Helpers for importing required and optional pyGWRx dependencies.

Imports remain lazy where that reduces startup cost, while error messages distinguish
between required runtime packages and feature-specific extras.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from importlib import import_module
from types import ModuleType
from typing import Optional


def import_required_dependency(
    module_name: str,
    *,
    purpose: Optional[str] = None,
) -> ModuleType:
    """Import a required runtime dependency or raise an actionable error.

    Args:
        module_name: Fully qualified module name to import.
        purpose: Optional description of the feature requiring the dependency.

    Returns:
        The imported module.

    Raises:
        ImportError: If the required dependency is unavailable.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        root_name = module_name.split(".", maxsplit=1)[0]
        if exc.name != root_name:
            raise
        context = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Required dependency {root_name!r} is unavailable{context}. "
            "Reinstall pyGWRx so its mandatory dependencies are installed."
        ) from exc


def import_optional_dependency(
    module_name: str,
    *,
    extra: str,
    purpose: Optional[str] = None,
) -> ModuleType:
    """Import an optional dependency or raise an actionable error.

    Args:
        module_name: Fully qualified module name to import.
        extra: pyGWRx extra that installs the dependency.
        purpose: Optional description of the feature requiring the dependency.

    Returns:
        The imported module.

    Raises:
        ImportError: If the requested optional dependency is unavailable.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        root_name = module_name.split(".", maxsplit=1)[0]
        if exc.name != root_name:
            raise
        context = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Optional dependency {root_name!r} is required{context}. "
            f'Install it with `pip install "pyGWRx[{extra}]"`.'
        ) from exc
