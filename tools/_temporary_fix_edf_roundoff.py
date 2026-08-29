from pathlib import Path

metrics_path = Path("src/pygwrx/core/metrics.py")
metrics = metrics_path.read_text(encoding="utf-8")
old = '''    trace_s_value = _validate_nonnegative_scalar(trace_S, "trace_S")
    trace_sts_value = _validate_nonnegative_scalar(trace_StS, "trace_StS")
    return float(n_value - 2.0 * trace_s_value + trace_sts_value)
'''
new = '''    trace_s_value = _validate_nonnegative_scalar(trace_S, "trace_S")
    trace_sts_value = _validate_nonnegative_scalar(trace_StS, "trace_StS")
    edf = float(n_value - 2.0 * trace_s_value + trace_sts_value)

    # A saturated smoother can have theoretical EDF == 0 while independent
    # floating-point trace calculations leave a tiny negative residue. Clamp only
    # machine-roundoff-scale negative zero; materially negative EDF values remain
    # negative so downstream validation still rejects an invalid diagnostic state.
    roundoff_tolerance = (
        16.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(n_value),
            2.0 * abs(trace_s_value),
            abs(trace_sts_value),
        )
    )
    if -roundoff_tolerance <= edf < 0.0:
        return 0.0
    return edf
'''
if old not in metrics:
    raise SystemExit("compute_edf target block not found")
metrics_path.write_text(metrics.replace(old, new, 1), encoding="utf-8", newline="\n")

test_path = Path("tests/test_metrics.py")
tests = test_path.read_text(encoding="utf-8")
tests = tests.replace(
    "from pygwrx.core import compute_aic, compute_aicc, compute_bic\n",
    "from pygwrx.core import compute_aic, compute_aicc, compute_bic, compute_edf\n",
    1,
)
addition = '''\n\ndef test_compute_edf_clamps_only_roundoff_scale_negative_zero():\n    trace_s = 40.0\n    trace_sts = 39.999999999999986\n    raw_edf = 40.0 - 2.0 * trace_s + trace_sts\n    assert raw_edf < 0.0\n    assert abs(raw_edf) < 1e-12\n    assert compute_edf(40, trace_s, trace_sts) == 0.0\n\n\ndef test_compute_edf_preserves_materially_negative_values():\n    assert compute_edf(40, trace_S=30.0, trace_StS=10.0) == pytest.approx(-10.0)\n'''
if "test_compute_edf_clamps_only_roundoff_scale_negative_zero" in tests:
    raise SystemExit("EDF tests already present")
test_path.write_text(tests.rstrip() + addition + "\n", encoding="utf-8", newline="\n")
