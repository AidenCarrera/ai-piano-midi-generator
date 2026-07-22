"""Analyze expressive event tokenization without writing training sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from symusic.core import ScoreSecond, ScoreTick

from ai_piano_midi_generator.tokenizer import (
    ExpressiveMidiTokenizer,
    TokenizerConfig,
)


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


def _percentile(sorted_values: list[float | int], percentage: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentage
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return float(
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def _statistics(values: list[float | int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
        "mean": round(mean(values), 6) if values else None,
        "median": round(median(values), 6) if values else None,
        "p90": round(value, 6)
        if (value := _percentile(ordered, 0.90)) is not None
        else None,
        "p95": round(value, 6)
        if (value := _percentile(ordered, 0.95)) is not None
        else None,
        "p99": round(value, 6)
        if (value := _percentile(ordered, 0.99)) is not None
        else None,
    }


def _token_family(token: str) -> str:
    if token.startswith("NOTE_ON_"):
        return "note_on"
    if token.startswith("NOTE_OFF_"):
        return "note_off"
    if token.startswith("VELOCITY_"):
        return "velocity"
    if token.startswith("TIME_SHIFT_"):
        return "time_shift"
    return token.casefold()


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _percentage(numerator: int, denominator: int) -> float | None:
    return round(numerator * 100.0 / denominator, 6) if denominator else None


def _markdown(report: dict[str, object]) -> str:
    settings = report["settings"]
    sequences = report["sequence_length_statistics"]
    timing = report["timing_error_seconds_statistics"]
    velocity = report["velocity_absolute_error_statistics"]
    retention = report["round_trip"]
    repairs = report["decoding_repairs"]
    pedal = report["pedal_analysis"]
    assert isinstance(settings, dict)
    assert isinstance(sequences, dict)
    assert isinstance(timing, dict)
    assert isinstance(velocity, dict)
    assert isinstance(retention, dict)
    assert isinstance(repairs, dict)
    assert isinstance(pedal, dict)

    lines = [
        "# Expressive MIDI tokenization analysis",
        "",
        "This report analyzes token streams in memory only. No tokenized training "
        "dataset was written.",
        "",
        "## Settings",
        "",
        f"- Files discovered: **{report['discovered_file_count']}**",
        f"- Files analyzed: **{report['analyzed_file_count']}**",
        f"- Vocabulary size: **{report['vocabulary_size']}**",
        f"- Timing resolution: **{settings['timing_resolution_seconds']} seconds**",
        "- Maximum time shift: "
        f"**{settings['effective_maximum_time_shift_seconds']} seconds**",
        f"- Velocity bins: **{settings['velocity_bin_count']}**",
        f"- Fixed decode tempo: **{settings['decode_tempo_qpm']} QPM**",
        "",
        "## Sequence lengths",
        "",
        f"- Total tokens: **{report['total_token_count']}**",
        f"- Minimum: **{sequences['min']}**",
        f"- Median: **{sequences['median']}**",
        f"- Mean: **{sequences['mean']}**",
        f"- P95: **{sequences['p95']}**",
        f"- P99: **{sequences['p99']}**",
        f"- Maximum: **{sequences['max']}**",
        "",
        "## Round-trip accuracy",
        "",
        "- Event-count retention: "
        f"**{retention['semantic_event_retention_percent']}%**",
        "- Decoded MIDI parseability: "
        f"**{retention['parseable_decoded_file_percent']}%**",
        f"- Mean absolute timing error: **{timing['mean']} seconds**",
        f"- P95 absolute timing error: **{timing['p95']} seconds**",
        f"- Maximum absolute timing error: **{float(timing['max']):.6f} seconds**",
        f"- Mean velocity error: **{velocity['mean']} MIDI units**",
        f"- Maximum velocity error: **{velocity['max']} MIDI units**",
        "- Minimum-duration encoding repairs: "
        f"**{report['minimum_duration_encoding_repairs']}**",
        f"- Decoding repairs: **{repairs['total']}**",
        "- Non-binary source CC64 values mapped to pedal states: "
        f"**{pedal['non_binary_value_event_count']}**",
        "- Redundant consecutive pedal-state events retained: "
        f"**{pedal['redundant_same_state_event_count']}**",
        "- Files ending with pedal down: "
        f"**{pedal['files_ending_with_pedal_down_count']}**",
        "",
        "## Semantic event distribution",
        "",
        "| Event | Count |",
        "| --- | ---: |",
    ]
    distribution = report["semantic_event_distribution"]
    assert isinstance(distribution, dict)
    for name, count in distribution.items():
        lines.append(f"| {name} | {count} |")

    lines.extend(["", "## Largest sequences", ""])
    largest = report["largest_sequences"]
    assert isinstance(largest, list)
    for item in largest:
        assert isinstance(item, dict)
        lines.append(f"- `{item['relative_path']}`: {item['token_count']} tokens")

    considerations = report["pre_training_considerations"]
    assert isinstance(considerations, list)
    lines.extend(["", "## Before dataset splitting and training", ""])
    for consideration in considerations:
        assert isinstance(consideration, dict)
        lines.append(f"- **{consideration['category']}:** {consideration['detail']}")

    issues = report["issues"]
    assert isinstance(issues, list)
    lines.extend(["", "## Issues", ""])
    if issues:
        for issue in issues:
            assert isinstance(issue, dict)
            lines.append(f"- `{issue['relative_path']}`: {issue['issue']}")
    else:
        lines.append("No tokenization or decoding issues were found.")
    lines.append("")
    return "\n".join(lines)


def analyze_tokenization(
    *,
    input_root: Path,
    report_json_path: Path,
    report_markdown_path: Path,
    config: TokenizerConfig,
) -> dict[str, object]:
    """Analyze all cleaned MIDI files without persisting token sequences."""
    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"MIDI input directory does not exist: {input_root}")
    files = sorted(
        (
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mid"
        ),
        key=lambda path: path.relative_to(input_root).as_posix().casefold(),
    )
    tokenizer = ExpressiveMidiTokenizer(config)

    sequence_lengths: list[int] = []
    timing_errors: list[float] = []
    velocity_errors: list[int] = []
    semantic_events: Counter[str] = Counter()
    token_families: Counter[str] = Counter()
    note_on_pitches: Counter[int] = Counter()
    velocity_bins: Counter[int] = Counter()
    time_shift_steps: Counter[int] = Counter()
    pedal_values: Counter[int] = Counter()
    repair_totals: Counter[str] = Counter()
    file_summaries: list[dict[str, object]] = []
    issues: list[dict[str, str]] = []
    minimum_duration_repairs = 0
    expected_semantic_events = 0
    decoded_semantic_events = 0
    decoded_parseable_files = 0
    event_count_match_files = 0
    redundant_pedal_states = 0
    files_starting_with_pedal_down: list[str] = []
    files_ending_with_pedal_down: list[str] = []

    for path in files:
        relative_path = path.relative_to(input_root).as_posix()
        try:
            source_score = ScoreSecond.from_file(path)
            encoded = tokenizer.encode(source_score)
            decoded = tokenizer.decode(encoded.tokens)
            reparsed = ScoreTick.from_midi(decoded.score.dumps_midi())
        except Exception as exc:
            issues.append(
                {
                    "relative_path": relative_path,
                    "issue": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        decoded_parseable_files += 1
        source_pedal_events = sorted(
            (float(control.time), int(control.value))
            for track in source_score.tracks
            for control in track.controls
            if control.number == 64
        )
        pedal_values.update(value for _, value in source_pedal_events)
        pedal_states = [value >= 64 for _, value in source_pedal_events]
        redundant_pedal_states += sum(
            previous == current
            for previous, current in zip(pedal_states, pedal_states[1:])
        )
        if pedal_states and pedal_states[0]:
            files_starting_with_pedal_down.append(relative_path)
        if pedal_states and pedal_states[-1]:
            files_ending_with_pedal_down.append(relative_path)
        sequence_lengths.append(len(encoded.tokens))
        timing_errors.extend(encoded.diagnostics.timing_errors_seconds)
        velocity_errors.extend(encoded.diagnostics.velocity_errors)
        semantic_events.update(encoded.diagnostics.event_counts)
        minimum_duration_repairs += encoded.diagnostics.minimum_duration_repairs
        for name, count in decoded.repairs.to_dict().items():
            if name != "total":
                repair_totals[name] += count

        for token in encoded.tokens:
            token_families[_token_family(token)] += 1
            if token.startswith("NOTE_ON_"):
                note_on_pitches[int(token.removeprefix("NOTE_ON_"))] += 1
            elif token.startswith("VELOCITY_"):
                velocity_bins[int(token.removeprefix("VELOCITY_"))] += 1
            elif token.startswith("TIME_SHIFT_"):
                time_shift_steps[int(token.removeprefix("TIME_SHIFT_"))] += 1

        source_event_count = sum(encoded.diagnostics.event_counts.values())
        decoded_note_count = sum(len(track.notes) for track in reparsed.tracks)
        decoded_cc64_count = sum(
            1
            for track in reparsed.tracks
            for control in track.controls
            if control.number == 64
        )
        decoded_event_count = decoded_note_count * 2 + decoded_cc64_count
        expected_semantic_events += source_event_count
        decoded_semantic_events += decoded_event_count
        if decoded_event_count == source_event_count:
            event_count_match_files += 1
        else:
            issues.append(
                {
                    "relative_path": relative_path,
                    "issue": (
                        f"semantic event count changed from {source_event_count} "
                        f"to {decoded_event_count}"
                    ),
                }
            )

        file_summaries.append(
            {
                "relative_path": relative_path,
                "token_count": len(encoded.tokens),
                "semantic_event_count": source_event_count,
                "maximum_timing_error_seconds": round(
                    max(encoded.diagnostics.timing_errors_seconds, default=0.0), 9
                ),
                "maximum_velocity_error": max(
                    encoded.diagnostics.velocity_errors, default=0
                ),
                "minimum_duration_encoding_repairs": (
                    encoded.diagnostics.minimum_duration_repairs
                ),
                "decoding_repairs": decoded.repairs.total,
            }
        )

    repair_report = dict(sorted(repair_totals.items()))
    repair_report["total"] = sum(repair_totals.values())
    pre_training_considerations = [
        {
            "category": "Sequence chunking",
            "detail": (
                f"Sequences reach {max(sequence_lengths, default=0)} tokens; choose "
                "a context/chunk length and carry active-note and pedal state across "
                "chunk boundaries."
            ),
        },
        {
            "category": "Sub-resolution notes",
            "detail": (
                f"{minimum_duration_repairs} notes were shorter than one timing step "
                "after quantization and were extended to one step."
            ),
        },
        {
            "category": "Binary sustain",
            "detail": (
                f"{sum(count for value, count in pedal_values.items() if value not in (0, 127))} "
                "non-binary CC64 values are represented as pedal down/up states, as "
                "required by this vocabulary."
            ),
        },
        {
            "category": "Pedal event efficiency",
            "detail": (
                f"{redundant_pedal_states} consecutive same-state pedal events are "
                "retained for event fidelity; decide whether to canonicalize them "
                "before training."
            ),
        },
        {
            "category": "Pedal boundaries",
            "detail": (
                f"{len(files_ending_with_pedal_down)} files end with pedal down; "
                "splitting must preserve or explicitly reset pedal state."
            ),
        },
    ]
    report: dict[str, object] = {
        "settings": config.to_dict(),
        "vocabulary_size": tokenizer.vocabulary_size,
        "discovered_file_count": len(files),
        "analyzed_file_count": len(file_summaries),
        "failed_file_count": len(files) - len(file_summaries),
        "total_token_count": sum(sequence_lengths),
        "sequence_length_statistics": _statistics(sequence_lengths),
        "timing_error_seconds_statistics": _statistics(timing_errors),
        "velocity_absolute_error_statistics": _statistics(velocity_errors),
        "timing_events_within_half_resolution_percent": _percentage(
            sum(
                error <= config.timing_resolution_seconds / 2 + 1e-12
                for error in timing_errors
            ),
            len(timing_errors),
        ),
        "timing_events_within_one_resolution_percent": _percentage(
            sum(
                error <= config.timing_resolution_seconds + 1e-12
                for error in timing_errors
            ),
            len(timing_errors),
        ),
        "exact_velocity_reconstruction_percent": _percentage(
            sum(error == 0 for error in velocity_errors), len(velocity_errors)
        ),
        "semantic_event_distribution": dict(sorted(semantic_events.items())),
        "token_family_distribution": dict(sorted(token_families.items())),
        "note_on_pitch_distribution": {
            str(pitch): count for pitch, count in sorted(note_on_pitches.items())
        },
        "velocity_bin_distribution": {
            str(velocity_bin): count
            for velocity_bin, count in sorted(velocity_bins.items())
        },
        "time_shift_step_distribution": {
            str(steps): count for steps, count in sorted(time_shift_steps.items())
        },
        "pedal_analysis": {
            "source_cc64_event_count": sum(pedal_values.values()),
            "source_value_distribution": {
                str(value): count for value, count in sorted(pedal_values.items())
            },
            "non_binary_value_event_count": sum(
                count for value, count in pedal_values.items() if value not in (0, 127)
            ),
            "redundant_same_state_event_count": redundant_pedal_states,
            "files_starting_with_pedal_down_count": len(files_starting_with_pedal_down),
            "files_starting_with_pedal_down": files_starting_with_pedal_down,
            "files_ending_with_pedal_down_count": len(files_ending_with_pedal_down),
            "files_ending_with_pedal_down": files_ending_with_pedal_down,
        },
        "minimum_duration_encoding_repairs": minimum_duration_repairs,
        "decoding_repairs": repair_report,
        "round_trip": {
            "expected_semantic_event_count": expected_semantic_events,
            "decoded_semantic_event_count": decoded_semantic_events,
            "semantic_event_retention_percent": _percentage(
                decoded_semantic_events, expected_semantic_events
            ),
            "event_count_match_file_count": event_count_match_files,
            "event_count_match_file_percent": _percentage(
                event_count_match_files, len(file_summaries)
            ),
            "parseable_decoded_file_count": decoded_parseable_files,
            "parseable_decoded_file_percent": _percentage(
                decoded_parseable_files, len(file_summaries)
            ),
        },
        "largest_sequences": sorted(
            file_summaries,
            key=lambda item: int(item["token_count"]),
            reverse=True,
        )[:20],
        "files_with_repairs": [
            item
            for item in file_summaries
            if item["minimum_duration_encoding_repairs"] or item["decoding_repairs"]
        ],
        "pre_training_considerations": pre_training_considerations,
        "issues": issues,
    }
    _write_atomic(
        report_json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_atomic(report_markdown_path, _markdown(report))
    return report


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
    assert isinstance(sequence_stats, dict)
    assert isinstance(timing_stats, dict)
    assert isinstance(velocity_stats, dict)
    assert isinstance(round_trip, dict)
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
    print(f"Decoding repairs: {report['decoding_repairs']['total']}")
    print(f"Issues: {len(report['issues'])}")
    return 1 if report["failed_file_count"] or report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
