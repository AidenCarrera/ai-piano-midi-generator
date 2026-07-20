"""CLI for conservative piano MIDI dataset preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ai_piano_midi_generator.data.cleaning import prepare_dataset, status_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select conservative piano tracks and write cleaned MIDI copies without "
            "modifying raw source files."
        )
    )
    parser.add_argument(
        "--raw-inventory",
        type=Path,
        default=Path("data/manifests/raw_inventory.jsonl"),
        help="Raw inspection inventory JSONL.",
    )
    parser.add_argument(
        "--track-selections",
        type=Path,
        default=Path("data/manifests/track_selections.jsonl"),
        help="User-maintained track selection overrides JSONL.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing immutable raw MIDI files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/midi"),
        help="Root directory for cleaned MIDI copies.",
    )
    parser.add_argument(
        "--processed-inventory",
        type=Path,
        default=Path("data/manifests/processed_inventory.jsonl"),
        help="Output cleaning inventory JSONL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing processed MIDI files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report decisions without writing MIDI or inventory files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = prepare_dataset(
            raw_inventory_path=args.raw_inventory,
            track_selections_path=args.track_selections,
            input_root=args.input_root,
            output_root=args.output_root,
            processed_inventory_path=args.processed_inventory,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    counts = status_counts(records)
    if args.dry_run:
        print("Dry run: no processed MIDI or inventory files were written.")
    print(f"Processed: {counts['processed']}")
    print(f"Needs review: {counts['needs_review']}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Failed: {counts['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
