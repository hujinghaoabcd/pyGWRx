# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Plain-text summary formatting for fitted pyGWRx objects.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _value_text(value: Any) -> str:
    """Render common scalar and array values compactly and deterministically."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_value_text(v)}" for k, v in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ", ".join(_value_text(item) for item in value) + "]"
    return str(value)


def format_summary(title: str, values: Mapping[str, Any], *, width: int = 78) -> str:
    """Return a terminal-friendly character table for a fitted object."""
    key_width = max([len(str(key)) for key in values] + [9])
    key_width = min(key_width, 28)
    inner = max(width - 4, key_width + 12)
    border = "+" + "-" * (inner + 2) + "+"
    heading = "| " + title.center(inner) + " |"
    separator = "+" + "=" * (inner + 2) + "+"
    lines = [border, heading, separator]
    value_width = inner - key_width - 3
    for key, value in values.items():
        text = _value_text(value)
        chunks = [
            text[i : i + value_width] for i in range(0, max(len(text), 1), value_width)
        ] or [""]
        lines.append(
            f"| {str(key)[:key_width]:<{key_width}} : {chunks[0]:<{value_width}} |"
        )
        for chunk in chunks[1:]:
            lines.append(f"| {'':<{key_width}}   {chunk:<{value_width}} |")
    lines.append(border)
    return "\n".join(lines)
