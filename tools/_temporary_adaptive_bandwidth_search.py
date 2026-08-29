from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


bandwidth_path = ROOT / "src" / "pygwrx" / "core" / "bandwidth.py"
text = bandwidth_path.read_text(encoding="utf-8")

# Adaptive search is now exhaustive over every integer k, so the sampled integer-grid
# helper is no longer part of the production path.
start = text.index("def _integer_grid(")
end = text.index("def _select_best_candidates(", start)
text = text[:start] + text[end:]

text = replace_once(
    text,
    '        self.verbose = _validate_bool(verbose, "verbose")\n\n',
    '        self.verbose = _validate_bool(verbose, "verbose")\n'
    '        self.search_trace_: tuple[tuple[Bandwidth, float], ...] = ()\n'
    '        self.best_score_: Optional[float] = None\n'
    '        self.search_range_: Optional[tuple[Bandwidth, Bandwidth]] = None\n\n',
    label="selector diagnostics attributes",
)

search_start = text.index("    def _search(\n", text.index("class _BaseSelector"))
search_end = text.index("    def _print_header", search_start)
new_search = '''    def _search(
        self,
        objective_raw: Callable[[Bandwidth], float],
        lower: Bandwidth,
        upper: Bandwidth,
    ) -> tuple[Bandwidth, float]:
        """Search the candidate domain and retain every evaluated score.

        Adaptive bandwidths are discrete neighbour orders, so every integer ``k`` in
        the validated range is evaluated exactly once. ``optimization_method`` and
        ``n_intervals`` continue to control only fixed-distance bandwidth searches.
        """
        cache: dict[Bandwidth, float] = {}
        self.search_trace_ = ()
        self.best_score_ = None
        self.search_range_ = (lower, upper)

        def objective(candidate: float) -> float:
            try:
                normalized = _normalize_candidate(
                    candidate,
                    adaptive=self.adaptive,
                    lower=lower,
                    upper=upper,
                )
            except _InvalidCandidateError:
                return np.inf

            if normalized not in cache:
                try:
                    score = float(objective_raw(normalized))
                except _InvalidCandidateError:
                    score = np.inf
                cache[normalized] = score if np.isfinite(score) else np.inf
            return cache[normalized]

        if self.adaptive:
            candidates = np.arange(int(lower), int(upper) + 1, dtype=int)
            best_bandwidth, best_score = _select_best_candidates(candidates, objective)
            self.search_trace_ = tuple(
                (int(candidate), float(cache[int(candidate)])) for candidate in candidates
            )
            self.best_score_ = float(best_score)
            return int(best_bandwidth), float(best_score)

        if self.optimization_method == "grid":
            candidates = np.linspace(float(lower), float(upper), self.n_intervals)
            best_bandwidth, best_score = _select_best_candidates(candidates, objective)

        elif self.optimization_method == "golden_section":
            from pygwrx.core.optimization import GoldenSectionSearch

            optimizer = GoldenSectionSearch(
                tol=1e-4,
                max_iter=100,
                verbose=self.verbose,
            )
            result = optimizer.minimize(
                objective,
                float(lower),
                float(upper),
                adaptive=False,
            )
            if not result.converged or not np.isfinite(result.score):
                raise RuntimeError("Golden-section bandwidth search did not converge.")

            candidate = _normalize_candidate(
                result.value,
                adaptive=False,
                lower=lower,
                upper=upper,
            )
            best_bandwidth = float(candidate)
            best_score = objective(best_bandwidth)
            if not np.isfinite(best_score):
                raise RuntimeError("Golden-section search returned an invalid bandwidth.")

        else:
            from pygwrx.core.optimization import BrentSearch

            optimizer = BrentSearch(tol=1e-5, max_iter=100, verbose=self.verbose)
            result = optimizer.minimize(objective, float(lower), float(upper))
            if not result.converged or not np.isfinite(result.score):
                raise RuntimeError("Brent bandwidth search did not converge.")

            candidate = _normalize_candidate(
                result.value,
                adaptive=False,
                lower=lower,
                upper=upper,
            )
            best_bandwidth = float(candidate)
            best_score = objective(best_bandwidth)
            if not np.isfinite(best_score):
                raise RuntimeError("Brent search returned an invalid bandwidth.")

        self.search_trace_ = tuple(
            (float(candidate), float(score))
            for candidate, score in sorted(cache.items(), key=lambda item: float(item[0]))
        )
        self.best_score_ = float(best_score)
        return float(best_bandwidth), float(best_score)

'''
text = text[:search_start] + new_search + text[search_end:]

old_header = '''        print(f"\\n{title}")
        print(f"  Method: {self.optimization_method}")
        if self.adaptive:
            print(f"  Search range: [{int(lower)}, {int(upper)}]")
            print("  Type: Adaptive (integer neighbour-order bandwidth)")
        else:
            print(f"  Search range: [{float(lower):.6g}, {float(upper):.6g}]")
            print("  Type: Fixed (distance bandwidth)")
'''
new_header = '''        print(f"\\n{title}")
        if self.adaptive:
            print("  Method: exhaustive_integer")
            print(f"  Search range: [{int(lower)}, {int(upper)}]")
            print("  Type: Adaptive (integer neighbour-order bandwidth)")
        else:
            print(f"  Method: {self.optimization_method}")
            print(f"  Search range: [{float(lower):.6g}, {float(upper):.6g}]")
            print("  Type: Fixed (distance bandwidth)")
'''
text = replace_once(text, old_header, new_header, label="verbose search header")

text = replace_once(
    text,
    '    """Select bandwidth by strict leave-one-out squared prediction error."""',
    '    """Select bandwidth by strict leave-one-out squared prediction error.\n\n'
    '    Adaptive searches evaluate every integer neighbour order in the validated\n'
    '    range and retain the ordered ``search_trace_`` after selection.\n'
    '    """',
    label="CV selector docstring",
)
text = replace_once(
    text,
    '    """Select bandwidth using Gaussian GWR AIC or AICc."""',
    '    """Select bandwidth using Gaussian GWR AIC or AICc.\n\n'
    '    Adaptive searches evaluate every integer neighbour order in the validated\n'
    '    range and retain the ordered ``search_trace_`` after selection.\n'
    '    """',
    label="AIC selector docstring",
)
text = replace_once(
    text,
    '    """Select bandwidth using Gaussian GWR BIC."""',
    '    """Select bandwidth using Gaussian GWR BIC.\n\n'
    '    Adaptive searches evaluate every integer neighbour order in the validated\n'
    '    range and retain the ordered ``search_trace_`` after selection.\n'
    '    """',
    label="BIC selector docstring",
)

bandwidth_path.write_text(text, encoding="utf-8")

# Focused unit tests: all adaptive optimization-method labels must resolve to the same
# exhaustive integer search, and invalid candidates remain visible in the trace.
test_path = ROOT / "tests" / "test_bandwidth_discrete_search.py"
test_path.write_text(
    '''"""Tests for exhaustive discrete adaptive-bandwidth selection."""

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
''',
    encoding="utf-8",
)

# Strengthen the existing synthetic external-reference test so n_intervals=2 cannot
# accidentally sample the adaptive domain. Every criterion must still recover the known
# shared-candidate external argmin and expose the complete integer trace.
reference_path = ROOT / "tests" / "test_gwr_external_references.py"
reference_text = reference_path.read_text(encoding="utf-8")
start = reference_text.index(
    "def test_controlled_adaptive_bandwidth_argmins_match_external_references("
)
end = reference_text.index("\ndef test_aicc_rejects_saturated_k4_boundary", start)
replacement = '''def test_controlled_adaptive_bandwidth_argmins_match_external_references(
    gwr_data: dict[str, Any],
) -> None:
    X_design = add_intercept(gwr_data["X"].to_numpy(dtype=float))
    y = gwr_data["y"].to_numpy(dtype=float)
    coords = gwr_data["coords"].to_numpy(dtype=float)
    kernel = get_kernel_function("bisquare")

    cv_selector = CrossValidationSelector(
        n_intervals=2,
        adaptive=True,
        optimization_method="golden_section",
    )
    aic_selector = AICSelector(
        n_intervals=2,
        corrected=False,
        adaptive=True,
        optimization_method="brent",
    )
    aicc_selector = AICSelector(
        n_intervals=2,
        corrected=True,
        adaptive=True,
        optimization_method="grid",
    )
    bic_selector = BICSelector(
        n_intervals=2,
        adaptive=True,
        optimization_method="golden_section",
    )

    cv = cv_selector.select(X_design, y, coords, kernel, bandwidth_range=(6, 40))
    aic = aic_selector.select(X_design, y, coords, kernel, bandwidth_range=(5, 40))
    aicc = aicc_selector.select(X_design, y, coords, kernel, bandwidth_range=(5, 40))
    bic = bic_selector.select(X_design, y, coords, kernel, bandwidth_range=(5, 40))

    # Shared-candidate external validation gives these same minima:
    # CV: PyGWRx = mgwr = GWmodel = 15
    # AIC: PyGWRx = mgwr = 5
    # AICc: PyGWRx = mgwr = GWmodel = 22
    # BIC: PyGWRx = mgwr = 5
    assert cv == 15
    assert aic == 5
    assert aicc == 22
    assert bic == 5

    assert tuple(k for k, _ in cv_selector.search_trace_) == tuple(range(6, 41))
    for selector in (aic_selector, aicc_selector, bic_selector):
        assert tuple(k for k, _ in selector.search_trace_) == tuple(range(5, 41))

'''
reference_text = reference_text[:start] + replacement + reference_text[end + 1 :]
reference_path.write_text(reference_text, encoding="utf-8")

# Real Columbus validation now drives the selectors themselves across the full raw k=4..49
# domain, including the near-saturated AICc boundary as +inf.
columbus_path = ROOT / "tests" / "test_gwr_columbus_reference.py"
columbus_text = columbus_path.read_text(encoding="utf-8")
columbus_text = replace_once(
    columbus_text,
    "from pygwrx.core.bandwidth import _fit_local_model, _kernel_weights\n",
    "from pygwrx.core.bandwidth import (\n"
    "    AICSelector,\n"
    "    CrossValidationSelector,\n"
    "    _fit_local_model,\n"
    "    _kernel_weights,\n"
    ")\n",
    label="Columbus selector imports",
)
start = columbus_text.index("def test_columbus_adaptive_bandwidth_argmins_are_stable(")
end = columbus_text.index("\ndef test_columbus_near_saturated_boundary_is_preserved", start)
replacement = '''def test_columbus_adaptive_bandwidth_argmins_are_stable(
    columbus_frame: pd.DataFrame,
) -> None:
    X_frame = columbus_frame[["INC", "HOVAL"]]
    y_series = columbus_frame["CRIME"]
    coords_frame = columbus_frame[["X", "Y"]]
    X_design = add_intercept(X_frame.to_numpy(dtype=float))
    y = y_series.to_numpy(dtype=float)
    coords = coords_frame.to_numpy(dtype=float)
    kernel = get_kernel_function("bisquare")

    cv_selector = CrossValidationSelector(
        n_intervals=2,
        adaptive=True,
        optimization_method="golden_section",
    )
    aicc_selector = AICSelector(
        n_intervals=2,
        corrected=True,
        adaptive=True,
        optimization_method="brent",
    )

    cv = cv_selector.select(
        X_design,
        y,
        coords,
        kernel,
        bandwidth_range=(4, 49),
    )
    aicc = aicc_selector.select(
        X_design,
        y,
        coords,
        kernel,
        bandwidth_range=(4, 49),
    )

    assert cv == 11
    assert aicc == 24
    assert tuple(k for k, _ in cv_selector.search_trace_) == tuple(range(4, 50))
    assert tuple(k for k, _ in aicc_selector.search_trace_) == tuple(range(4, 50))
    assert np.isinf(dict(aicc_selector.search_trace_)[4])

    summary = _load("bandwidth_summary.json")
    assert summary["criteria"]["cv_sse"]["pygwrx_raw_argmin"] == cv
    assert summary["criteria"]["aicc"]["pygwrx_k_ge_5_argmin"] == aicc

'''
columbus_text = columbus_text[:start] + replacement + columbus_text[end + 1 :]
columbus_path.write_text(columbus_text, encoding="utf-8")

# Show the retained trace in the maintained public example.
example_path = ROOT / "examples" / "core" / "07_bandwidth_selectors.py"
example_text = example_path.read_text(encoding="utf-8")
old_loop = '''for selector in selectors:
    print(
        type(selector).__name__,
        selector.select(Xa, ya, ca, gaussian_kernel, bandwidth_range=(10, 18)),
    )
'''
new_loop = '''for selector in selectors:
    selected = selector.select(
        Xa,
        ya,
        ca,
        gaussian_kernel,
        bandwidth_range=(10, 18),
    )
    print(
        type(selector).__name__,
        selected,
        "evaluated=",
        len(selector.search_trace_),
    )
'''
example_text = replace_once(example_text, old_loop, new_loop, label="bandwidth example")
example_path.write_text(example_text, encoding="utf-8")

# Keep the public test-baseline text accurate. PRs #19 and #20 raised the pre-change suite
# to 418 tests; this PR adds four non-reference parameterized/unit cases, for 422 total.
for readme_name in ("README.md", "README.zh.md"):
    readme_path = ROOT / readme_name
    readme = readme_path.read_text(encoding="utf-8")
    readme = replace_once(readme, "406 passed", "422 passed", label=readme_name)
    readme_path.write_text(readme, encoding="utf-8")

reference_readme_path = ROOT / "tools" / "reference" / "gwr" / "README.md"
reference_readme = reference_readme_path.read_text(encoding="utf-8")
reference_readme = replace_once(
    reference_readme,
    "45 tests marked `reference`, separate from 361 non-reference tests",
    "45 tests marked `reference`, separate from 377 non-reference tests",
    label="reference-suite counts",
)
reference_readme_path.write_text(reference_readme, encoding="utf-8")
