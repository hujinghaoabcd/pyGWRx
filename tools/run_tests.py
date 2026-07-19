# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run the non-reference test suite in isolated, time-bounded batches."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_BATCH_COUNT = 3
_DEFAULT_TIMEOUT_SECONDS = 600


def _test_files() -> list[Path]:
    """Return all maintained pytest modules in deterministic order."""
    return sorted(Path("tests").glob("test_*.py"))


def _batch_files(batch: int) -> list[Path]:
    """Return one balanced alphabetical partition of the maintained tests."""
    files = _test_files()
    quotient, remainder = divmod(len(files), _BATCH_COUNT)
    sizes = [quotient + int(index < remainder) for index in range(_BATCH_COUNT)]
    start = sum(sizes[: batch - 1])
    selected = files[start : start + sizes[batch - 1]]
    if not selected:
        raise RuntimeError(f"Test batch {batch} contains no tests.")
    return selected


def run_batch(batch: int, timeout_seconds: int) -> None:
    """Run one non-reference batch in a fresh pytest process."""
    selected = _batch_files(batch)
    env = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        "not reference",
        *(str(path) for path in selected),
    ]
    print(f"Running test batch {batch}/{_BATCH_COUNT}:", flush=True)
    for path in selected:
        print(f"  - {path}", flush=True)
    try:
        subprocess.run(command, check=True, env=env, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Test batch {batch} exceeded {timeout_seconds} seconds."
        ) from exc


def main() -> None:
    """Parse command-line arguments and execute one test batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", type=int, choices=range(1, _BATCH_COUNT + 1), required=True
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Maximum runtime for the pytest subprocess.",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    run_batch(args.batch, args.timeout_seconds)


if __name__ == "__main__":
    main()
