#!/usr/bin/env python
"""Entry point: build the book, run the model, provision it, report it.

Usage
-----
    python scripts/run_pipeline.py                      # synthetic book (default)
    python scripts/run_pipeline.py --data-mode real     # LendingClub extract in data/raw/loan.csv
    python scripts/run_pipeline.py --loans 250000       # larger synthetic book
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditrisk.report import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the credit risk / IFRS 9 pipeline.")
    parser.add_argument(
        "--data-mode",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic (reproducible, no download) or real (LendingClub extract required)",
    )
    parser.add_argument("--loans", type=int, default=120_000, help="number of synthetic loans to generate")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    arguments = parser.parse_args()

    run_pipeline(
        data_mode=arguments.data_mode,
        n_loans=arguments.loans,
        verbose=not arguments.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
