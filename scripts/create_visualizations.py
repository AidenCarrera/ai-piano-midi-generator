"""Create dataset and experiment figures from existing project reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ai_piano_midi_generator.visualization import generate_project_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing reports/ and data/manifests/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg", "both"),
        default="png",
        help="Figure format to save.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping a plot whose source report is unavailable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_directory = args.output
    if not output_directory.is_absolute():
        output_directory = project_root / output_directory
    formats = ("png", "svg") if args.format == "both" else (args.format,)
    result = generate_project_figures(
        project_root,
        output_directory,
        formats=formats,
        dpi=args.dpi,
        strict=args.strict,
    )
    print(
        json.dumps(
            {
                "generated_files": [
                    path.relative_to(project_root).as_posix()
                    if path.is_relative_to(project_root)
                    else path.as_posix()
                    for path in result.generated_files
                ],
                "skipped_plots": list(result.skipped_plots),
            },
            indent=2,
        )
    )
    return 0 if result.generated_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
