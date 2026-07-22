"""Final validation for the cleaned piano MIDI dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from symusic.core import ScoreSecond, ScoreTick

from ai_piano_midi_generator.data.cleaning import load_raw_inventory


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path.resolve()}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path} line {line_number}")
        records.append(value)
    return records


def _numeric_stats(values: Sequence[float | int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": round(mean(values), 6) if values else None,
    }


def _record_list(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key, [])
    return value if isinstance(value, list) else []


def _record_status(record: dict[str, object]) -> str:
    status = record.get("status")
    return status if isinstance(status, str) else "invalid"


def _integer_field(record: dict[str, object], key: str) -> int:
    value = record.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _relative_path(record: dict[str, object]) -> str:
    value = record.get("relative_path")
    if not isinstance(value, str):
        raise ValueError("Processed inventory record is missing relative_path")
    return Path(value.replace("\\", "/")).as_posix()


def _expects_output(record: dict[str, object]) -> bool:
    status = _record_status(record)
    if status == "processed":
        return True
    warnings = _record_list(record, "warnings")
    return status == "skipped" and "output_exists_use_overwrite" in warnings


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _markdown_report(report: dict[str, object]) -> str:
    statuses = report["run_status_counts"]
    assert isinstance(statuses, dict)
    lines = [
        "# Final cleaned dataset validation",
        "",
        f"- Validation passed: **{report['validation_passed']}**",
        f"- Raw files: **{report['raw_file_count']}**",
        f"- Final usable files: **{report['final_usable_file_count']}**",
        "",
        "## Preparation run",
        "",
        "| Status | Files |",
        "| --- | ---: |",
        f"| Processed | {statuses['processed']} |",
        f"| Rejected | {statuses['rejected']} |",
        f"| Needs review | {statuses['needs_review']} |",
        f"| Skipped | {statuses['skipped']} |",
        f"| Failed | {statuses['failed']} |",
        "",
        "## Dataset totals",
        "",
        f"- Raw notes: **{report['raw_note_count']}**",
        f"- Processed notes: **{report['processed_note_count']}**",
        f"- Invalid notes removed: **{report['invalid_notes_removed']}**",
        f"- Discarded-track notes: **{report['discarded_track_note_count']}**",
        f"- Selected source tracks: **{report['selected_track_count']}**",
        f"- Discarded source tracks: **{report['discarded_track_count']}**",
        "- Files with merged piano tracks: "
        f"**{report['multiple_merged_piano_file_count']}**",
        f"- Files with CC64 events: **{report['files_with_cc64_count']}**",
        f"- Remaining invalid notes: **{report['remaining_invalid_note_count']}**",
        "",
        "## Integrity checks",
        "",
        f"- All expected outputs present: **{report['all_expected_outputs_present']}**",
        f"- All outputs parseable: **{report['all_outputs_parseable']}**",
        f"- Output filenames unique: **{report['output_filenames_unique']}**",
        f"- Outputs traceable to sources: **{report['outputs_traceable']}**",
        "",
        "## Rejected files",
        "",
    ]
    rejected = report["rejected_files"]
    assert isinstance(rejected, list)
    if rejected:
        for item in rejected:
            assert isinstance(item, dict)
            lines.append(f"- `{item['relative_path']}` - {item['reason']}")
    else:
        lines.append("No files were rejected.")

    lines.extend(["", "## Missing or suspicious outputs", ""])
    suspicious = report["suspicious_outputs"]
    assert isinstance(suspicious, dict)
    if any(suspicious.values()):
        lines.append("```json")
        lines.append(json.dumps(suspicious, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("No missing or suspicious outputs were found.")
    lines.append("")
    return "\n".join(lines)


def validate_final_dataset(
    *,
    raw_inventory_path: Path,
    processed_inventory_path: Path,
    output_root: Path,
    report_json_path: Path,
    report_markdown_path: Path,
) -> dict[str, object]:
    """Parse every expected output and write final statistical/integrity reports."""
    raw_records = load_raw_inventory(raw_inventory_path)
    processed_records = _read_jsonl(processed_inventory_path)
    raw_paths = {_relative_path(record) for record in raw_records}
    processed_paths = [_relative_path(record) for record in processed_records]
    if len(processed_paths) != len(set(processed_paths)):
        raise ValueError("Processed inventory contains duplicate relative paths")

    status_counter = Counter(_record_status(record) for record in processed_records)
    expected_paths = {
        _relative_path(record)
        for record in processed_records
        if _expects_output(record)
    }
    output_root = output_root.resolve()
    actual_files = sorted(
        (
            path
            for path in output_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mid"
        ),
        key=lambda path: path.relative_to(output_root).as_posix().casefold(),
    )
    actual_by_relative = {
        path.relative_to(output_root).as_posix(): path for path in actual_files
    }
    actual_paths = set(actual_by_relative)

    missing_outputs = sorted(expected_paths - actual_paths)
    unexpected_outputs = sorted(actual_paths - expected_paths)
    untraceable_outputs = sorted(actual_paths - raw_paths)
    basename_groups: dict[str, list[str]] = defaultdict(list)
    for relative_path in actual_paths:
        basename_groups[Path(relative_path).name.casefold()].append(relative_path)
    duplicate_filenames = [
        sorted(paths) for paths in basename_groups.values() if len(paths) > 1
    ]

    parse_errors: list[dict[str, str]] = []
    remaining_invalid_notes: list[dict[str, object]] = []
    pitches: list[int] = []
    velocities: list[int] = []
    note_durations_seconds: list[float] = []
    file_durations_seconds: list[float] = []
    files_with_cc64: list[str] = []
    suspicious_pitch_outputs: list[dict[str, object]] = []
    total_cc64_events = 0
    parsed_output_count = 0

    for relative_path in sorted(expected_paths & actual_paths):
        path = actual_by_relative[relative_path]
        try:
            score_tick = ScoreTick.from_file(path)
            score_second = ScoreSecond.from_file(path)
            parsed_output_count += 1
        except Exception as exc:
            parse_errors.append(
                {
                    "relative_path": relative_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        invalid_count = 0
        cc64_count = 0
        pitch_outliers: list[int] = []
        for track in score_tick.tracks:
            cc64_count += sum(1 for control in track.controls if control.number == 64)
            for note in track.notes:
                pitches.append(note.pitch)
                velocities.append(note.velocity)
                if note.pitch < 21 or note.pitch > 108:
                    pitch_outliers.append(note.pitch)
                if note.duration <= 0:
                    invalid_count += 1
        for track in score_second.tracks:
            note_durations_seconds.extend(note.duration for note in track.notes)
        file_durations_seconds.append(score_second.end())
        if cc64_count:
            files_with_cc64.append(relative_path)
            total_cc64_events += cc64_count
        if invalid_count:
            remaining_invalid_notes.append(
                {"relative_path": relative_path, "count": invalid_count}
            )
        if pitch_outliers:
            suspicious_pitch_outputs.append(
                {
                    "relative_path": relative_path,
                    "count": len(pitch_outliers),
                    "min": min(pitch_outliers),
                    "max": max(pitch_outliers),
                }
            )

    rejected_files = [
        {
            "relative_path": _relative_path(record),
            "reason": record.get("rejection_reason") or "No reason recorded",
        }
        for record in processed_records
        if _record_status(record) == "rejected"
    ]
    failed_files = [
        {
            "relative_path": _relative_path(record),
            "error": record.get("error"),
        }
        for record in processed_records
        if _record_status(record) == "failed"
    ]
    needs_review_files = [
        {
            "relative_path": _relative_path(record),
            "warnings": _record_list(record, "warnings"),
        }
        for record in processed_records
        if _record_status(record) == "needs_review"
    ]
    merged_files = [
        _relative_path(record)
        for record in processed_records
        if len(_record_list(record, "selected_tracks")) > 1
    ]

    suspicious_outputs = {
        "missing_expected_outputs": missing_outputs,
        "unexpected_outputs": unexpected_outputs,
        "untraceable_outputs": untraceable_outputs,
        "duplicate_output_filenames": duplicate_filenames,
        "parse_errors": parse_errors,
        "pitches_outside_88_key_piano_range": suspicious_pitch_outputs,
    }
    remaining_invalid_count = sum(
        _integer_field(item, "count") for item in remaining_invalid_notes
    )
    validation_passed = not any(
        [
            missing_outputs,
            unexpected_outputs,
            untraceable_outputs,
            duplicate_filenames,
            parse_errors,
            remaining_invalid_notes,
            failed_files,
            needs_review_files,
        ]
    )

    raw_note_count = sum(_integer_field(record, "note_count") for record in raw_records)
    selected_track_count = sum(
        len(_record_list(record, "selected_tracks")) for record in processed_records
    )
    discarded_tracks = [
        track
        for record in processed_records
        for track in _record_list(record, "removed_tracks")
        if isinstance(track, dict)
    ]
    invalid_notes_removed = sum(
        len(_record_list(record, "invalid_notes_removed"))
        for record in processed_records
    )

    report: dict[str, object] = {
        "validation_passed": validation_passed,
        "run_status_counts": {
            "processed": status_counter["processed"],
            "rejected": status_counter["rejected"],
            "needs_review": status_counter["needs_review"],
            "skipped": status_counter["skipped"],
            "failed": status_counter["failed"],
        },
        "raw_file_count": len(raw_records),
        "expected_output_file_count": len(expected_paths),
        "actual_output_file_count": len(actual_paths),
        "final_usable_file_count": parsed_output_count,
        "raw_note_count": raw_note_count,
        "processed_note_count": len(pitches),
        "invalid_notes_removed": invalid_notes_removed,
        "discarded_track_note_count": sum(
            _integer_field(track, "note_count") for track in discarded_tracks
        ),
        "selected_track_count": selected_track_count,
        "discarded_track_count": len(discarded_tracks),
        "multiple_merged_piano_file_count": len(merged_files),
        "files_with_multiple_merged_piano_tracks": merged_files,
        "files_with_cc64_count": len(files_with_cc64),
        "files_with_cc64_sustain_events": files_with_cc64,
        "total_cc64_sustain_events": total_cc64_events,
        "pitch_statistics": _numeric_stats(pitches),
        "velocity_statistics": _numeric_stats(velocities),
        "note_duration_seconds_statistics": _numeric_stats(note_durations_seconds),
        "file_duration_seconds_statistics": _numeric_stats(file_durations_seconds),
        "remaining_invalid_note_count": remaining_invalid_count,
        "remaining_invalid_notes": remaining_invalid_notes,
        "out_of_88_key_piano_range_note_count": sum(
            _integer_field(item, "count") for item in suspicious_pitch_outputs
        ),
        "rejected_files": rejected_files,
        "failed_files": failed_files,
        "needs_review_files": needs_review_files,
        "all_expected_outputs_present": not missing_outputs,
        "all_outputs_parseable": not parse_errors,
        "output_filenames_unique": not duplicate_filenames,
        "outputs_traceable": not untraceable_outputs,
        "suspicious_outputs": suspicious_outputs,
    }
    _write_atomic(
        report_json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_atomic(report_markdown_path, _markdown_report(report))
    return report
