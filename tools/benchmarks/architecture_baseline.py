"""Generate observational performance/memory baselines for architecture refactors.

The measurements in this module are intentionally *not* CI timing gates.  They
capture representative execution profiles before the 0.2 architecture rewrite
so later pull requests can compare wall time, memory, and retained dense state
without changing numerical definitions merely to satisfy a benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import pygwrx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "architecture_baseline.json"


def _regression_data(seed: int, n: int, p: int = 2):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    times = np.linspace(0.0, 8.0, n)
    X = rng.normal(size=(n, p))
    coefficients = np.linspace(1.4, -0.6, p)
    spatial_effect = 0.12 * coords[:, 0] - 0.05 * coords[:, 1]
    temporal_effect = 0.03 * times
    y = 1.0 + X @ coefficients + spatial_effect + temporal_effect
    y += rng.normal(0.0, 0.12, size=n)
    return X, y, coords, times


def _fit_gwr_manual_streaming():
    X, y, coords, _ = _regression_data(401, 500, 3)
    model = pygwrx.GWR(kernel="gaussian", bandwidth=3.0, adaptive=False)
    model.fit(
        X,
        y,
        coords,
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
    )
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "manual fixed bandwidth; streamed GWR fit; hat disabled",
        "checks": {"hat_matrix_is_none": model.hat_matrix_ is None},
    }


def _fit_gwr_manual_hat():
    X, y, coords, _ = _regression_data(402, 500, 3)
    model = pygwrx.GWR(kernel="gaussian", bandwidth=3.0, adaptive=False)
    model.fit(
        X,
        y,
        coords,
        compute_hat_matrix=True,
        compute_local_r2=False,
        compute_inference=False,
    )
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "manual fixed bandwidth; full GWR hat retained",
        "checks": {"hat_matrix_shape": list(model.hat_matrix_.shape)},
    }


def _fit_gwr_auto_bandwidth():
    X, y, coords, _ = _regression_data(403, 120, 2)
    model = pygwrx.GWR(
        kernel="gaussian",
        bandwidth="cv",
        adaptive=True,
        bandwidth_range=(30, 45),
    )
    model.fit(
        X,
        y,
        coords,
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
    )
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "automatic adaptive CV bandwidth selection + streamed final fit",
        "checks": {
            "selected_bandwidth": int(model.bandwidth_),
            "inside_requested_range": 30 <= int(model.bandwidth_) <= 45,
        },
    }


def _fit_mgwr_backfit():
    X, y, coords, _ = _regression_data(404, 100, 2)
    bandwidths = [45] * (X.shape[1] + 1)
    model = pygwrx.MGWR(
        bandwidths=bandwidths,
        adaptive=True,
        init_bandwidth=45,
        max_iter=3,
        tol=1e-4,
        verbose=False,
    )
    model.fit(X, y, coords, compute_inference=False)
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "representative multiscale backfit with repeated-use distances",
        "checks": {
            "bandwidth_count": int(len(model.bandwidths_)),
            "iterations": int(model.n_iter_),
        },
    }


def _fit_gtwr():
    X, y, coords, times = _regression_data(405, 160, 2)
    model = pygwrx.GTWR(
        kernel="gaussian",
        bandwidth=4.0,
        lambda_st=0.5,
        causal=False,
    )
    model.fit(
        X,
        y,
        coords,
        times,
        compute_local_r2=False,
        compute_inference=False,
        compute_hat_matrix=False,
    )
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "representative fixed-bandwidth spatiotemporal fit",
        "checks": {"hat_matrix_is_none": model.hat_matrix_ is None},
    }


def _fit_mgtwr_small():
    X, y, coords, times = _regression_data(406, 80, 2)
    model = pygwrx.MGTWR(
        bandwidths=[4.0, 3.5, 4.5],
        taus=[0.7, 1.0, 0.6],
        kernel="gaussian",
        adaptive=False,
        calculate_inference=False,
        max_iter=3,
        tol_multi=1e-4,
    )
    model.fit(X, y, coords, times)
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "small representative multiscale spatiotemporal backfit",
        "checks": {
            "bandwidth_count": int(len(model.bandwidths_)),
            "tau_count": int(len(model.taus_)),
        },
    }


def _fit_sgwr_weight_heavy():
    X, y, coords, _ = _regression_data(407, 300, 3)
    model = pygwrx.SGWR(
        bandwidth=3.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.4,
        store_weights=True,
    )
    model.fit(X, y, coords)
    checks = {}
    for name in ("spatial_weights_", "similarity_weights_", "combined_weights_"):
        value = getattr(model, name, None)
        checks[f"{name}_shape"] = list(value.shape) if value is not None else None
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "weight-heavy SGWR fit with dense weight retention enabled",
        "checks": checks,
    }


def _fit_scalable_gwr_large_n():
    X, y, coords, _ = _regression_data(408, 2500, 3)
    model = pygwrx.ScalableGWR(
        bandwidth=100,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=0.05,
        verbose=False,
    )
    model.fit(X, y, coords)
    return model, {
        "n": len(y),
        "p": X.shape[1],
        "path": "large-n cKDTree/compressed ScaGWR fit",
        "checks": {
            "coefficient_rows": int(model.coefficients_.shape[0]),
            "training_rows": int(model.X_train_.shape[0]),
        },
    }


SCENARIOS: dict[str, Callable[[], tuple[Any, dict[str, Any]]]] = {
    "gwr_manual_streaming": _fit_gwr_manual_streaming,
    "gwr_manual_hat": _fit_gwr_manual_hat,
    "gwr_auto_bandwidth": _fit_gwr_auto_bandwidth,
    "mgwr_backfit": _fit_mgwr_backfit,
    "gtwr_fit": _fit_gtwr,
    "mgtwr_small": _fit_mgtwr_small,
    "sgwr_weight_heavy": _fit_sgwr_weight_heavy,
    "scalable_gwr_large_n": _fit_scalable_gwr_large_n,
}


def _peak_rss_mb() -> float | None:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return None
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _dense_square_inventory(model: Any, n: int) -> list[dict[str, Any]]:
    arrays = []
    for name, value in vars(model).items():
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape == (n, n):
            arrays.append(
                {
                    "attribute": name,
                    "shape": [n, n],
                    "megabytes": value.nbytes / (1024.0 * 1024.0),
                }
            )
    return sorted(arrays, key=lambda item: item["attribute"])


def _fingerprint(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("fitted_values_", "coef_", "coefficients_", "params_"):
        value = getattr(model, name, None)
        if isinstance(value, np.ndarray):
            result[f"{name}_sum"] = float(np.sum(value))
            result[f"{name}_shape"] = list(value.shape)
    bandwidth = getattr(model, "bandwidth_", None)
    if isinstance(bandwidth, (int, float, np.integer, np.floating)):
        result["bandwidth_"] = float(bandwidth)
    bandwidths = getattr(model, "bandwidths_", None)
    if isinstance(bandwidths, np.ndarray):
        result["bandwidths_"] = np.asarray(bandwidths, dtype=float).tolist()
    return result


def _run_worker(name: str) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown benchmark scenario: {name}")

    gc.collect()
    rss_before = _peak_rss_mb()
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    model, metadata = SCENARIOS[name]()
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _peak_rss_mb()

    n = int(metadata["n"])
    dense_arrays = _dense_square_inventory(model, n)
    return {
        "scenario": name,
        "n": n,
        "p": int(metadata["p"]),
        "path": metadata["path"],
        "metrics": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "python_peak_megabytes": python_peak / (1024.0 * 1024.0),
            "process_peak_rss_megabytes": rss_after,
            "process_peak_rss_delta_megabytes": (
                None
                if rss_before is None or rss_after is None
                else max(0.0, rss_after - rss_before)
            ),
            "retained_dense_square_megabytes": sum(
                item["megabytes"] for item in dense_arrays
            ),
        },
        "retained_dense_square_arrays": dense_arrays,
        "checks": metadata["checks"],
        "fingerprint": _fingerprint(model),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment() -> dict[str, Any]:
    import scipy

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _run_parent(names: list[str], output: Path) -> dict[str, Any]:
    results = {}
    for name in names:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", name]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        results[name] = json.loads(completed.stdout)

    document = {
        "schema_version": 1,
        "purpose": (
            "Observational pre-refactor baseline only; wall time and memory values "
            "are not CI pass/fail thresholds."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "environment": _environment(),
        "scenario_order": names,
        "scenarios": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS))
    parser.add_argument("--list", action="store_true", dest="list_scenarios")
    parser.add_argument("--worker", choices=sorted(SCENARIOS), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_scenarios:
        print(json.dumps(sorted(SCENARIOS)))
        return
    if args.worker is not None:
        print(json.dumps(_run_worker(args.worker), sort_keys=True))
        return

    names = args.scenario or list(SCENARIOS)
    document = _run_parent(names, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scenario_count": len(document["scenarios"]),
                "git_sha": document["git_sha"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
