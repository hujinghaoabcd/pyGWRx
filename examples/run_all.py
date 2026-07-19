# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run every public pyGWRx example in isolated Python processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CATEGORIES = ("models", "core", "diagnostics", "io", "plotting", "workflows")


@dataclass(frozen=True)
class _ExampleResult:
    """Result from one isolated example process."""

    path: Path
    returncode: int
    output: str
    timed_out: bool = False


def scripts() -> list[Path]:
    """Return maintained example scripts in stable documentation order."""
    return [
        path
        for category in CATEGORIES
        for path in sorted((EXAMPLES / category).glob("*.py"))
        if not path.name.startswith("_")
    ]


def _example_environment() -> dict[str, str]:
    """Build the deterministic environment shared by example subprocesses."""
    env = os.environ.copy()
    pythonpath = [str(ROOT / "src"), str(EXAMPLES), str(EXAMPLES / "plotting")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _run_script(path: Path, env: dict[str, str], timeout: int) -> _ExampleResult:
    """Run one example and capture its combined output."""
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return _ExampleResult(path, 124, output, timed_out=True)
    return _ExampleResult(path, completed.returncode, completed.stdout)


def main() -> None:
    """Run all examples with bounded process concurrency and report failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Maximum number of isolated example processes to run concurrently.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-example timeout in seconds.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1.")

    env = _example_environment()
    all_scripts = scripts()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_run_script, path, env, args.timeout)
            for path in all_scripts
        ]
        results = [future.result() for future in futures]

    failures: list[str] = []
    for index, result in enumerate(results, start=1):
        relative = result.path.relative_to(ROOT)
        print(f"[{index:02d}/{len(all_scripts):02d}] {relative}", flush=True)
        if result.returncode:
            reason = "timed out" if result.timed_out else f"exit {result.returncode}"
            failures.append(f"{relative} ({reason})")
            if result.output:
                print(result.output)
        else:
            tail = result.output.strip().splitlines()[-1:] or ["completed"]
            print(f"       {tail[0]}")

    if failures:
        raise SystemExit("Failed examples:\n- " + "\n- ".join(failures))
    print(f"All {len(all_scripts)} examples completed successfully.")


if __name__ == "__main__":
    main()
