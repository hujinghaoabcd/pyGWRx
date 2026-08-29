"""Contract tests for the non-gating A4 architecture benchmark harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
HARNESS = ROOT / "tools" / "benchmarks" / "architecture_baseline.py"
BASELINE = ROOT / "benchmarks" / "architecture_baseline.json"
REQUIRED_SCENARIOS = {
    "gwr_manual_streaming",
    "gwr_auto_bandwidth",
    "mgwr_backfit",
    "gtwr_fit",
    "mgtwr_small",
    "sgwr_weight_heavy",
    "scalable_gwr_large_n",
}


def _run_live_scenario(name: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(HARNESS), "--worker", name],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_harness_exposes_required_architecture_scenarios():
    completed = subprocess.run(
        [sys.executable, str(HARNESS), "--list"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    scenarios = set(json.loads(completed.stdout))
    assert REQUIRED_SCENARIOS <= scenarios
    assert "gwr_manual_hat" in scenarios


def test_committed_baseline_is_observational_not_a_timing_gate():
    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert "not CI pass/fail thresholds" in document["purpose"]
    assert REQUIRED_SCENARIOS <= set(document["scenarios"])
    assert set(document["scenario_order"]) == set(document["scenarios"])

    forbidden = {
        "max_wall_seconds",
        "max_memory_mb",
        "timing_threshold",
        "fail_if_slow",
    }
    assert not forbidden.intersection(document)

    for name, result in document["scenarios"].items():
        assert result["scenario"] == name
        assert result["n"] > 0
        assert result["p"] > 0
        assert result["path"]
        assert result["fingerprint"]
        metrics = result["metrics"]
        assert metrics["wall_seconds"] >= 0.0
        assert metrics["cpu_seconds"] >= 0.0
        assert metrics["python_peak_megabytes"] >= 0.0
        assert metrics["retained_dense_square_megabytes"] >= 0.0


def test_committed_structural_execution_markers_are_documented():
    scenarios = json.loads(BASELINE.read_text(encoding="utf-8"))["scenarios"]

    assert scenarios["gwr_manual_streaming"]["checks"]["hat_matrix_is_none"] is True
    assert scenarios["gwr_manual_streaming"]["retained_dense_square_buffers"] == []

    hat_buffers = scenarios["gwr_manual_hat"]["retained_dense_square_buffers"]
    assert len(hat_buffers) == 1
    assert set(hat_buffers[0]["attributes"]) == {"S_matrix_", "hat_matrix_"}

    assert scenarios["scalable_gwr_large_n"]["retained_dense_square_buffers"] == []

    sgwr_buffers = scenarios["sgwr_weight_heavy"]["retained_dense_square_buffers"]
    assert len(sgwr_buffers) >= 3


def test_live_runtime_preserves_structural_memory_paths():
    streamed = _run_live_scenario("gwr_manual_streaming")
    assert streamed["checks"]["hat_matrix_is_none"] is True
    assert streamed["retained_dense_square_buffers"] == []

    with_hat = _run_live_scenario("gwr_manual_hat")
    hat_buffers = with_hat["retained_dense_square_buffers"]
    assert len(hat_buffers) == 1
    assert set(hat_buffers[0]["attributes"]) == {"S_matrix_", "hat_matrix_"}

    scalable = _run_live_scenario("scalable_gwr_large_n")
    assert scalable["n"] >= 2_000
    assert scalable["retained_dense_square_buffers"] == []
