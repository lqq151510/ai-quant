#!/usr/bin/env python3
"""Time the capital-scale study so performance claims can be reproduced.

Each run starts a fresh interpreter and writes into a temporary directory; no
research artifact is created or overwritten. Imports are part of the measured
wall time because that is what a user waits for.

Usage:
    python3 tools/bench_capital_scale.py
    python3 tools/bench_capital_scale.py --runs 5 --signals momentum reverse equal_weight
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "research_v1" / ".venv" / "bin" / "python"
SNAPSHOT = ROOT / "research_v1" / "artifacts" / "market-etf-verified-20261108"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="timed runs, default 3")
    parser.add_argument("--signals", nargs="+", default=None,
                        help="optional signal subset passed to capital_scale")
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise SystemExit(f"missing research virtualenv: {PYTHON}")
    if not SNAPSHOT.is_dir():
        raise SystemExit(f"missing audited snapshot: {SNAPSHOT}")

    timings = []
    for index in range(args.runs):
        workdir = Path(tempfile.mkdtemp(prefix="aiquant-bench-"))
        try:
            cmd = [str(PYTHON), "-m", "quant_research.capital_scale",
                   "--snapshot", str(SNAPSHOT), "--out", str(workdir / "out")]
            if args.signals:
                cmd += ["--signals", *args.signals]
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            started = time.monotonic()
            completed = subprocess.run(cmd, cwd=ROOT / "research_v1", env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            elapsed = time.monotonic() - started
            if completed.returncode != 0:
                raise SystemExit(f"run {index + 1} failed:\n{completed.stderr[-2000:]}")
            timings.append(elapsed)
            print(f"run {index + 1}: {elapsed:.2f}s")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"median: {statistics.median(timings):.2f}s   min: {min(timings):.2f}s   runs: {len(timings)}")


if __name__ == "__main__":
    main()
