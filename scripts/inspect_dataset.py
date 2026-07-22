"""Command-line entry point for read-only raw MIDI dataset inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ai_piano_midi_generator.data.inspection import inspect_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively inspect raw .mid files without modifying them, then write "
            "a JSONL inventory and JSON/Markdown summaries."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing raw MIDI files (searched recursively).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/raw_inventory.jsonl"),
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/dataset/inventory.json"),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=Path("reports/dataset/inventory.md"),
        help="Output Markdown summary path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = inspect_dataset(
            input_dir=args.input,
            manifest_path=args.manifest,
            summary_json_path=args.summary_json,
            summary_markdown_path=args.summary_markdown,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    counts = summary["classification_counts"]
    assert isinstance(counts, dict)
    print(f"MIDI files: {summary['total_midi_files']}")
    print(f"Accepted: {counts['accepted']}")
    print(f"Needs review: {counts['needs_review']}")
    print(f"Rejected: {counts['rejected']}")
    print(f"Manifest: {args.manifest}")
    print(f"JSON summary: {args.summary_json}")
    print(f"Markdown summary: {args.summary_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
