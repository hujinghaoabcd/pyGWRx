"""Tests for exhaustive discrete adaptive-bandwidth selection."""

from __future__ import annotations

import numpy as np
import pytest

from pygwrx.core.bandwidth import CrossValidationSelector, _InvalidCandidateError


@pytest.mark.parametrize("method", ["grid", "golden_section", "brent"])
def test_adaptive_search_exhaustively_evaluates_every_integer(method: str) -> None:
    selector = CrossValidationSelector(
        n_intervals=2,
        optimization_method=method,
        adaptive=True,
    )
    scores = {4: 9.0, 5: 5.0, 6: 7.0, 7: 0.5, 8: 3.0, 9: 2.0}

    best, score = selector._search(lambda k: scores[int(k)], 4, 9)

    assert best == 7
    assert score == pytest.approx(0.5)
    assert selector.best_score_ == pytest.approx(0.5)
    assert selector.search_range_ == (4, 9)
    assert selector.search_trace_ == tuple((k, scores[k]) for k in range(4, 10))


def test_adaptive_search_trace_preserves_invalid_candidates() -> None:
    selector = CrossValidationSelector(
        n_intervals=2,
        optimization_method="golden_section",
        adaptive=True,
    )

    def objective(k: int) -> float:
        if int(k) == 5:
            raise _InvalidCandidateError("boundary candidate is not estimable")
        return float((int(k) - 7) ** 2)

    best, score = selector._search(objective, 4, 8)
    trace = dict(selector.search_trace_)

    assert best == 7
    assert score == pytest.approx(0.0)
    assert tuple(trace) == (4, 5, 6, 7, 8)
    assert np.isinf(trace[5])
