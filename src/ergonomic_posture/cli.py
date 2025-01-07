from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ergonomic posture assessment and recommendation demo pipeline."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/demo"))
    parser.add_argument("--subjects", type=int, default=14)
    parser.add_argument("--windows-per-subject", type=int, default=72)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_demo(
        output_dir=args.output_dir,
        n_subjects=args.subjects,
        windows_per_subject=args.windows_per_subject,
    )
    print(json.dumps(result["best_model"], indent=2))


if __name__ == "__main__":
    main()
