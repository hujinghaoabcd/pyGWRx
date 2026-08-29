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

    forbidden = {"max_wall_seconds", "max_memory_mb", "timing_threshold", "fail_if_slow"}
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


def test_structural_execution_markers_are_preserved_in_baseline():
    scenarios = json.loads(BASELINE.read_text(encoding="utf-8"))["scenarios"]

    assert scenarios["gwr_manual_streaming"]["checks"]["hat_matrix_is_none"] is True
    assert scenarios["gwr_manual_streaming"]["retained_dense_square_arrays"] == []

    hat_arrays = {
        item["attribute"]
        for item in scenarios["gwr_manual_hat"]["retained_dense_square_arrays"]
    }
    assert "hat_matrix_" in hat_arrays or "S_matrix_" in hat_arrays

    assert scenarios["scalable_gwr_large_n"]["retained_dense_square_arrays"] == []

    sgwr_arrays = scenarios["sgwr_weight_heavy"]["retained_dense_square_arrays"]
    assert len(sgwr_arrays) >= 2
