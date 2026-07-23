"""Analyze expressive event tokenization without writing training sequences."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ai_piano_midi_generator.tokenization_analysis import analyze_tokenization
from ai_piano_midi_generator.tokenizer import TokenizerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze expressive MIDI tokenization and round-trip decoding without "
            "writing a tokenized training dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/midi"),
        help="Root containing cleaned MIDI files.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/tokenization/analysis.json"),
        help="Machine-readable analysis report.",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/tokenization/analysis.md"),
        help="Human-readable analysis report.",
    )
    parser.add_argument(
        "--timing-resolution",
        type=float,
        default=0.01,
        help="Elapsed seconds represented by one timing step.",
    )
    parser.add_argument(
        "--maximum-time-shift",
        type=float,
        default=1.0,
        help="Maximum elapsed seconds represented by one time-shift token.",
    )
    parser.add_argument(
        "--velocity-bins",
        type=int,
        default=32,
        help="Number of velocity bins.",
    )
    parser.add_argument(
        "--decode-tempo",
        type=float,
        default=120.0,
        help="Fixed decoding tempo in quarter notes per minute.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = TokenizerConfig(
            timing_resolution_seconds=args.timing_resolution,
            maximum_time_shift_seconds=args.maximum_time_shift,
            velocity_bin_count=args.velocity_bins,
            decode_tempo_qpm=args.decode_tempo,
        )
        report = analyze_tokenization(
            input_root=args.input,
            report_json_path=args.report_json,
            report_markdown_path=args.report_markdown,
            config=config,
        )
    except (NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    sequence_stats = report["sequence_length_statistics"]
    timing_stats = report["timing_error_seconds_statistics"]
    velocity_stats = report["velocity_absolute_error_statistics"]
    round_trip = report["round_trip"]
    decoding_repairs = report["decoding_repairs"]
    issues = report["issues"]
    assert isinstance(sequence_stats, dict)
    assert isinstance(timing_stats, dict)
    assert isinstance(velocity_stats, dict)
    assert isinstance(round_trip, dict)
    assert isinstance(decoding_repairs, dict)
    assert isinstance(issues, list)
    print(f"Files analyzed: {report['analyzed_file_count']}")
    print(f"Vocabulary size: {report['vocabulary_size']}")
    print(f"Total tokens: {report['total_token_count']}")
    print(
        "Sequence lengths: "
        f"median={sequence_stats['median']}, p95={sequence_stats['p95']}, "
        f"max={sequence_stats['max']}"
    )
    print(
        "Timing error (seconds): "
        f"mean={timing_stats['mean']}, max={timing_stats['max']}"
    )
    print(f"Velocity error: mean={velocity_stats['mean']}, max={velocity_stats['max']}")
    print(
        f"Semantic event retention: {round_trip['semantic_event_retention_percent']}%"
    )
    print(f"Decoding repairs: {decoding_repairs['total']}")
    print(f"Issues: {len(issues)}")
    return 1 if report["failed_file_count"] or report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
