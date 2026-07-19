# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run branch coverage in isolated test-file batches and combine the results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_BATCH_ENDS = (11, 21)
_COVERAGE_FILES = tuple(Path(f".coverage.batch{index}") for index in range(1, 4))
_DEFAULT_TIMEOUT_SECONDS = 600


def _test_files() -> list[Path]:
    """Return maintained test modules included in package coverage."""
    files = sorted(Path("tests").glob("test_*.py"))
    return [path for path in files if path.name != "test_examples_standalone.py"]


def _batch_files(batch: int) -> list[Path]:
    """Return one of three stable alphabetical test-file partitions."""
    files = _test_files()
    boundaries = (0, *_BATCH_ENDS, len(files))
    selected = files[boundaries[batch - 1] : boundaries[batch]]
    if not selected:
        raise RuntimeError(f"Coverage batch {batch} contains no tests.")
    return selected


def run_batch(batch: int, timeout_seconds: int) -> None:
    """Run one time-bounded coverage batch in a fresh data file."""
    target = _COVERAGE_FILES[batch - 1]
    target.unlink(missing_ok=True)
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(target)
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    env["MPLBACKEND"] = "Agg"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        "not reference",
        *(str(path) for path in _batch_files(batch)),
        "--cov=pygwrx",
        "--cov-report=",
        "--cov-fail-under=0",
    ]
    try:
        subprocess.run(command, check=True, env=env, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Coverage batch {batch} exceeded {timeout_seconds} seconds."
        ) from exc


def combine() -> None:
    """Combine batch data, enforce the configured threshold, and write XML."""
    missing = [str(path) for path in _COVERAGE_FILES if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing coverage batch files: {', '.join(missing)}")
    Path(".coverage").unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "combine",
            *(str(path) for path in _COVERAGE_FILES),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--fail-under=74"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "coverage", "xml", "-o", "coverage.xml"],
        check=True,
    )


def main() -> None:
    """Parse command-line arguments and run or combine coverage batches."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=int, choices=(1, 2, 3))
    group.add_argument("--combine", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Maximum runtime for one coverage pytest subprocess.",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    if args.combine:
        combine()
    else:
        run_batch(args.batch, args.timeout_seconds)


if __name__ == "__main__":
    main()
