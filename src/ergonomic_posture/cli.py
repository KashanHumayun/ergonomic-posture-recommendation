from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ergonomic posture assessment pipeline on the real UCI postural-transitions dataset."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
    print(json.dumps({"best_model": result["best_model"]}, indent=2))


if __name__ == "__main__":
    main()
