"""Read-only MIDI dataset inspection and reporting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from symusic.core import ScoreSecond

Classification = Literal["accepted", "needs_review", "rejected"]

PIANO_PROGRAMS = frozenset(range(8))


@dataclass(slots=True)
class FileInspection:
    """Serializable inspection result for one MIDI file."""

    filename: str
    relative_path: str
    sha256: str | None = None
    parse_success: bool = False
    error: str | None = None
    classification: Classification = "rejected"
    duration_seconds: float | None = None
    track_count: int = 0
    track_names: list[str] = field(default_factory=list)
    midi_programs: list[int] = field(default_factory=list)
    drum_or_percussion_tracks: list[dict[str, object]] = field(default_factory=list)
    track_note_counts: list[dict[str, object]] = field(default_factory=list)
    note_count: int = 0
    pitch_range: dict[str, int] | None = None
    velocity_range: dict[str, int] | None = None
    sustain_pedal_cc64_event_count: int = 0
    tempo_event_count: int = 0
    time_signature_changes: list[dict[str, object]] = field(default_factory=list)
    invalid_or_zero_length_notes: list[dict[str, object]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def discover_midi_files(input_dir: Path) -> list[Path]:
    """Recursively find MIDI files using a case-insensitive ``.mid`` suffix."""
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mid"
    ]
    return sorted(
        files,
        key=lambda path: (path.relative_to(input_dir).as_posix().casefold(), str(path)),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the entire payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as midi_file:
        for chunk in iter(lambda: midi_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _note_issues(
    *, time: float, duration: float, pitch: int, velocity: int
) -> list[str]:
    issues: list[str] = []
    if duration <= 0:
        issues.append("zero_or_negative_duration")
    if time < 0:
        issues.append("negative_start_time")
    if not 0 <= pitch <= 127:
        issues.append("pitch_out_of_range")
    if not 1 <= velocity <= 127:
        issues.append("velocity_out_of_range")
    return issues


def _classify(
    *,
    note_count: int,
    valid_pitched_note_count: int,
    invalid_note_count: int,
    drum_track_count: int,
    non_piano_programs: set[int],
) -> tuple[Classification, list[str]]:
    issues: list[str] = []
    if note_count == 0:
        issues.append("no_notes")
    if note_count > 0 and valid_pitched_note_count == 0:
        issues.append("no_valid_pitched_notes")
    if invalid_note_count:
        issues.append("invalid_or_zero_length_notes")
    if drum_track_count:
        issues.append("drum_or_percussion_tracks")
    if non_piano_programs:
        issues.append("non_piano_programs")

    if note_count == 0 or valid_pitched_note_count == 0:
        return "rejected", issues
    if issues:
        return "needs_review", issues
    return "accepted", issues


def inspect_midi_file(path: Path, input_dir: Path) -> FileInspection:
    """Inspect one MIDI file, returning a rejected record on any read/parse error."""
    input_dir = input_dir.resolve()
    path = path.resolve()
    relative_path = path.relative_to(input_dir).as_posix()
    result = FileInspection(filename=path.name, relative_path=relative_path)

    try:
        result.sha256 = sha256_file(path)
        score = ScoreSecond.from_file(path)

        pitches: list[int] = []
        velocities: list[int] = []
        invalid_notes: list[dict[str, object]] = []
        programs: set[int] = set()
        non_piano_programs: set[int] = set()
        valid_pitched_note_count = 0

        result.parse_success = True
        result.duration_seconds = round(score.end(), 6)
        result.track_count = len(score.tracks)
        result.track_names = [track.name for track in score.tracks]
        result.tempo_event_count = len(score.tempos)
        result.time_signature_changes = [
            {
                "time_seconds": round(event.time, 6),
                "numerator": event.numerator,
                "denominator": event.denominator,
            }
            for event in score.time_signatures
        ]

        for track_index, track in enumerate(score.tracks):
            track_note_count = len(track.notes)
            result.note_count += track_note_count
            if track_note_count:
                programs.add(track.program)
                if not track.is_drum and track.program not in PIANO_PROGRAMS:
                    non_piano_programs.add(track.program)

            result.track_note_counts.append(
                {
                    "track_index": track_index,
                    "name": track.name,
                    "note_count": track_note_count,
                }
            )

            if track.is_drum:
                result.drum_or_percussion_tracks.append(
                    {
                        "track_index": track_index,
                        "name": track.name,
                        "program": track.program,
                        "note_count": track_note_count,
                    }
                )

            result.sustain_pedal_cc64_event_count += sum(
                1 for control in track.controls if control.number == 64
            )

            for note_index, note in enumerate(track.notes):
                pitch = note.pitch
                velocity = note.velocity
                time = note.time
                duration = note.duration
                pitches.append(pitch)
                velocities.append(velocity)
                reasons = _note_issues(
                    time=time,
                    duration=duration,
                    pitch=pitch,
                    velocity=velocity,
                )
                if reasons:
                    invalid_notes.append(
                        {
                            "track_index": track_index,
                            "track_name": track.name,
                            "note_index": note_index,
                            "time_seconds": round(time, 6),
                            "duration_seconds": round(duration, 6),
                            "pitch": pitch,
                            "velocity": velocity,
                            "reasons": reasons,
                        }
                    )
                elif not track.is_drum:
                    valid_pitched_note_count += 1

        result.midi_programs = sorted(programs)
        result.invalid_or_zero_length_notes = invalid_notes
        if pitches:
            result.pitch_range = {"min": min(pitches), "max": max(pitches)}
        if velocities:
            result.velocity_range = {
                "min": min(velocities),
                "max": max(velocities),
            }

        result.classification, result.issues = _classify(
            note_count=result.note_count,
            valid_pitched_note_count=valid_pitched_note_count,
            invalid_note_count=len(invalid_notes),
            drum_track_count=len(result.drum_or_percussion_tracks),
            non_piano_programs=non_piano_programs,
        )
    except Exception as exc:  # malformed files must not stop the dataset scan
        result.parse_success = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.classification = "rejected"
        result.issues = ["parse_error"]

    return result


def _build_summary(
    input_dir: Path, records: list[FileInspection], generated_at: str
) -> dict[str, object]:
    classifications = Counter(record.classification for record in records)
    issue_counts = Counter(issue for record in records for issue in record.issues)
    program_counts: Counter[int] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)

    for record in records:
        program_counts.update(record.midi_programs)
        if record.sha256 is not None:
            hashes[record.sha256].append(record.relative_path)

    duplicate_groups = [
        {"sha256": digest, "files": paths}
        for digest, paths in sorted(hashes.items())
        if len(paths) > 1
    ]
    durations = [
        record.duration_seconds
        for record in records
        if record.duration_seconds is not None
    ]

    return {
        "generated_at": generated_at,
        "input_directory": str(input_dir.resolve()),
        "total_midi_files": len(records),
        "parse_success_count": sum(record.parse_success for record in records),
        "parse_error_count": sum(not record.parse_success for record in records),
        "classification_counts": {
            "accepted": classifications["accepted"],
            "needs_review": classifications["needs_review"],
            "rejected": classifications["rejected"],
        },
        "total_notes": sum(record.note_count for record in records),
        "total_sustain_pedal_cc64_events": sum(
            record.sustain_pedal_cc64_event_count for record in records
        ),
        "total_tempo_events": sum(record.tempo_event_count for record in records),
        "total_time_signature_changes": sum(
            len(record.time_signature_changes) for record in records
        ),
        "total_invalid_or_zero_length_notes": sum(
            len(record.invalid_or_zero_length_notes) for record in records
        ),
        "duration_seconds": {
            "min": round(min(durations), 6) if durations else None,
            "max": round(max(durations), 6) if durations else None,
            "mean": round(mean(durations), 6) if durations else None,
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "midi_program_file_counts": {
            str(program): count for program, count in sorted(program_counts.items())
        },
        "duplicate_content_group_count": len(duplicate_groups),
        "duplicate_content_groups": duplicate_groups,
        "rejected_files": [
            {
                "relative_path": record.relative_path,
                "error": record.error,
                "issues": record.issues,
            }
            for record in records
            if record.classification == "rejected"
        ],
    }


def _markdown_summary(summary: dict[str, object]) -> str:
    counts = summary["classification_counts"]
    assert isinstance(counts, dict)
    issue_counts = summary["issue_counts"]
    assert isinstance(issue_counts, dict)
    duration = summary["duration_seconds"]
    assert isinstance(duration, dict)
    rejected_files = summary["rejected_files"]
    assert isinstance(rejected_files, list)

    lines = [
        "# Dataset inspection summary",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Input: `{summary['input_directory']}`",
        f"- MIDI files: **{summary['total_midi_files']}**",
        f"- Parsed successfully: **{summary['parse_success_count']}**",
        f"- Parse errors: **{summary['parse_error_count']}**",
        "",
        "## Classification",
        "",
        "| Classification | Files |",
        "| --- | ---: |",
        f"| Accepted | {counts['accepted']} |",
        f"| Needs review | {counts['needs_review']} |",
        f"| Rejected | {counts['rejected']} |",
        "",
        "Accepted files parse successfully and contain valid pitched notes using only",
        "General MIDI piano programs 0-7, with no drum tracks or invalid notes.",
        "Files needing review contain usable pitched notes but also contain drums,",
        "non-piano programs, or invalid/zero-length notes. Files are rejected when",
        "they cannot be parsed or contain no valid pitched notes.",
        "",
        "## Aggregate statistics",
        "",
        f"- Notes: **{summary['total_notes']}**",
        f"- Sustain-pedal CC64 events: **{summary['total_sustain_pedal_cc64_events']}**",
        f"- Tempo events: **{summary['total_tempo_events']}**",
        f"- Time-signature changes: **{summary['total_time_signature_changes']}**",
        "- Invalid or zero-length notes: "
        f"**{summary['total_invalid_or_zero_length_notes']}**",
        "- Duration range (seconds): "
        f"**{duration['min']}** to **{duration['max']}** "
        f"(mean **{duration['mean']}**)",
        f"- Duplicate-content groups: **{summary['duplicate_content_group_count']}**",
        "",
        "## Issues",
        "",
    ]

    if issue_counts:
        lines.extend(["| Issue | Files |", "| --- | ---: |"])
        lines.extend(
            f"| `{issue}` | {count} |" for issue, count in issue_counts.items()
        )
    else:
        lines.append("No file-level issues were detected.")

    lines.extend(["", "## Rejected files", ""])
    if rejected_files:
        for rejected in rejected_files:
            assert isinstance(rejected, dict)
            detail = rejected.get("error") or ", ".join(rejected["issues"])
            lines.append(f"- `{rejected['relative_path']}` — {detail}")
    else:
        lines.append("No files were rejected.")

    lines.append("")
    return "\n".join(lines)


def _ensure_outputs_outside_input(input_dir: Path, output_paths: list[Path]) -> None:
    input_dir = input_dir.resolve()
    for output_path in output_paths:
        if output_path.resolve().is_relative_to(input_dir):
            raise ValueError(
                f"Refusing to write inspection output inside raw input: {output_path}"
            )


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def inspect_dataset(
    input_dir: Path,
    manifest_path: Path,
    summary_json_path: Path,
    summary_markdown_path: Path,
) -> dict[str, object]:
    """Inspect a directory tree and write the inventory and aggregate reports."""
    input_dir = input_dir.resolve()
    output_paths = [manifest_path, summary_json_path, summary_markdown_path]
    _ensure_outputs_outside_input(input_dir, output_paths)
    midi_files = discover_midi_files(input_dir)
    records = [inspect_midi_file(path, input_dir) for path in midi_files]
    generated_at = datetime.now(UTC).isoformat()
    summary = _build_summary(input_dir, records, generated_at)

    manifest = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _write_text_atomic(manifest_path, manifest)
    _write_text_atomic(
        summary_json_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(summary_markdown_path, _markdown_summary(summary))
    return summary
