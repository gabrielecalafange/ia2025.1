"""
Helper script to run incremental training sequentially across multiple dataset phases.
"""

import argparse
from pathlib import Path

from train_incremental import DATA_DIR, MODEL_PATH, run_incremental_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate incremental training across multiple dataset phases."
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        help="Specific phase numbers to run (overrides start/end).",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=2,
        help="First phase to run when --phases is not provided (default: 2).",
    )
    parser.add_argument(
        "--end-phase",
        type=int,
        default=5,
        help="Last phase to run when --phases is not provided (default: 5).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory that stores the phase CSV files (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the incremental model package (default: {MODEL_PATH}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.phases:
        phases = args.phases
    else:
        if args.end_phase < args.start_phase:
            raise ValueError("end-phase must be >= start-phase")
        phases = list(range(args.start_phase, args.end_phase + 1))

    for phase in phases:
        data_path = args.data_dir / f"email_spam_phase{phase}.csv"
        print(f"\n=== Incremental training for phase {phase} ===")
        run_incremental_update(data_path, args.model_path)


if __name__ == "__main__":
    main()
