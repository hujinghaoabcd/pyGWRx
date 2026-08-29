from pathlib import Path

p = Path("src/pygwrx/core/solver.py")
s = p.read_text(encoding="utf-8")
old = '''        ridge: Non-negative regularization shared with ``compute_hat_matrix``.

    Returns:
        local_coefs: Local coefficient vectors.

    Notes:
        If a location has fewer positive-weight observations than design-matrix columns,
        the function emits a warning and returns the deterministic ridge-regularized
        solution. It never copies coefficients from a preceding location and never falls
        back silently to global OLS, so results do not depend on target ordering.
'''
new = '''        ridge: Optional non-negative ridge penalty. The default ``0.0`` is standard
            unpenalized WLS; positive values are explicit lower-level regularization.

    Returns:
        local_coefs: Local coefficient vectors.

    Notes:
        If a location has fewer positive-weight observations than design-matrix columns,
        the default path warns and returns the Moore-Penrose minimum-norm unpenalized WLS
        solution. It never copies coefficients from a preceding location and never falls
        back silently to global OLS, so results do not depend on target ordering.
'''
if old not in s:
    raise SystemExit("stale local_regression wording not found")
s = s.replace(old, new, 1)
old_hat = '''        ridge: Non-negative regularization. The same value and normal-system construction are
            used by ``weighted_least_squares`` and ``local_regression``.
'''
new_hat = '''        ridge: Optional non-negative ridge penalty. The default ``0.0`` is unpenalized;
            positive values explicitly regularize the standalone smoother calculation.
'''
if old_hat not in s:
    raise SystemExit("stale compute_hat_matrix wording not found")
s = s.replace(old_hat, new_hat, 1)
p.write_text(s, encoding="utf-8", newline="\n")

p = Path("src/pygwrx/core/bandwidth.py")
b = p.read_text(encoding="utf-8")
old_comment = '''# A single numerical regularization value is used for both coefficient fitting and
# hat-matrix calculations, so the fitted values and trace(S) refer to the same smoother.
_RIDGE = 0.0
'''
new_comment = '''# Standard GWR bandwidth scoring is unpenalized. The constant is retained internally
# only to make that numerical policy explicit at the local-solver call site.
_RIDGE = 0.0
'''
if old_comment not in b:
    raise SystemExit("stale bandwidth ridge comment not found")
b = b.replace(old_comment, new_comment, 1)
p.write_text(b, encoding="utf-8", newline="\n")

p = Path("CHANGELOG.md")
c = p.read_text(encoding="utf-8")
c = c.replace(
    "- Aligned GWR calibration, prediction, and bandwidth scoring on the same unpenalized WLS semantics.",
    "- Aligned standard GWR calibration, target prediction, and bandwidth scoring on the same unpenalized WLS semantics.",
    1,
)
p.write_text(c, encoding="utf-8", newline="\n")
