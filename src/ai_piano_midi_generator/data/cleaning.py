"""Conservative, raw-safe preparation of piano-only MIDI files."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from symusic.core import ScoreTick, TrackTick

from ai_piano_midi_generator.data.inspection import sha256_file

CleaningStatus = Literal["processed", "rejected", "needs_review", "skipped", "failed"]

PIANO_PROGRAMS = frozenset(range(8))
EXCLUDED_TRACK_NAME_WORDS = frozenset(
    {
        "bass",
        "cello",
        "choir",
        "clarinet",
        "drum",
        "drums",
        "flute",
        "guitar",
        "organ",
        "percussion",
        "sax",
        "saxophone",
        "strings",
        "trombone",
        "trumpet",
        "viola",
        "violin",
        "vocal",
        "vocals",
        "voice",
    }
)


@dataclass(frozen=True, slots=True)
class TrackSelectionOverride:
    """Explicit user decision for one source file."""

    relative_path: str
    selected_track_indices: tuple[int, ...] | None = None
    cc64_track_indices: tuple[int, ...] = ()
    merge: bool = False
    skip: bool = False
    reject: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Result of conservative track selection."""

    selected_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    cc64_indices: tuple[int, ...]
    merge: bool
    needs_review: bool
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class CleaningRecord:
    """Serializable outcome for one raw MIDI file."""

    relative_path: str
    source_path: str
    output_path: str
    status: CleaningStatus
    selected_tracks: list[dict[str, object]] = field(default_factory=list)
    cc64_source_tracks: list[dict[str, object]] = field(default_factory=list)
    removed_tracks: list[dict[str, object]] = field(default_factory=list)
    notes_kept: int = 0
    pedal_events_kept: int = 0
    invalid_notes_removed: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    rejection_reason: str | None = None
    dry_run: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        result = asdict(self)
        result["invalid_notes_removed_count"] = len(self.invalid_notes_removed)
        return result


def _normalized_relative_path(value: str) -> str:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def _load_jsonl(path: Path) -> list[tuple[int, dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path.resolve()}")

    rows: list[tuple[int, dict[str, object]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path} line {line_number}")
        rows.append((line_number, value))
    return rows


def load_raw_inventory(path: Path) -> list[dict[str, object]]:
    """Load and minimally validate the raw inspection inventory."""
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, row in _load_jsonl(path):
        relative_path = row.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError(
                f"Missing string relative_path in {path} line {line_number}"
            )
        normalized = _normalized_relative_path(relative_path)
        if normalized in seen:
            raise ValueError(f"Duplicate relative_path in raw inventory: {normalized}")
        seen.add(normalized)
        row["relative_path"] = normalized
        records.append(row)
    return records


def load_track_selections(path: Path) -> dict[str, TrackSelectionOverride]:
    """Load user-maintained per-file track selections from JSONL."""
    overrides: dict[str, TrackSelectionOverride] = {}
    for line_number, row in _load_jsonl(path):
        relative_path = row.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError(
                f"Missing string relative_path in {path} line {line_number}"
            )
        normalized = _normalized_relative_path(relative_path)
        if normalized in overrides:
            raise ValueError(f"Duplicate track selection override: {normalized}")

        skip = row.get("skip", False)
        reject = row.get("reject", False)
        merge = row.get("merge", False)
        selected_value = row.get("selected_track_indices")
        cc64_value = row.get("cc64_track_indices", [])
        reason = row.get("reason")
        if (
            not isinstance(skip, bool)
            or not isinstance(reject, bool)
            or not isinstance(merge, bool)
        ):
            raise ValueError(
                f"skip, reject, and merge must be booleans in {path} line {line_number}"
            )
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"reason must be a string in {path} line {line_number}")

        selected_indices: tuple[int, ...] | None = None
        if selected_value is not None:
            if not isinstance(selected_value, list) or not all(
                isinstance(index, int) and not isinstance(index, bool)
                for index in selected_value
            ):
                raise ValueError(
                    "selected_track_indices must be a list of integers in "
                    f"{path} line {line_number}"
                )
            selected_indices = tuple(selected_value)
            if not selected_indices:
                raise ValueError(
                    f"selected_track_indices cannot be empty in {path} line {line_number}"
                )
            if len(selected_indices) != len(set(selected_indices)):
                raise ValueError(f"Duplicate selected track index in {normalized}")

        if not isinstance(cc64_value, list) or not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in cc64_value
        ):
            raise ValueError(
                f"cc64_track_indices must be a list of integers in {path} line {line_number}"
            )
        cc64_indices = tuple(cc64_value)
        if len(cc64_indices) != len(set(cc64_indices)):
            raise ValueError(f"Duplicate CC64 track index in {normalized}")

        if skip and reject:
            raise ValueError(
                f"Override cannot be both skipped and rejected: {normalized}"
            )
        if (skip or reject) and (selected_indices is not None or cc64_indices):
            raise ValueError(
                f"Skipped or rejected override cannot select tracks: {normalized}"
            )
        if reject and (reason is None or not reason.strip()):
            raise ValueError(f"Rejected override requires a reason: {normalized}")
        if not skip and not reject and selected_indices is None:
            raise ValueError(
                "Override must select tracks, set skip=true, or set reject=true: "
                f"{normalized}"
            )
        if merge and (selected_indices is None or len(selected_indices) < 2):
            raise ValueError(
                f"merge=true requires multiple selected tracks: {normalized}"
            )

        overrides[normalized] = TrackSelectionOverride(
            relative_path=normalized,
            selected_track_indices=selected_indices,
            cc64_track_indices=cc64_indices,
            merge=merge,
            skip=skip,
            reject=reject,
            reason=reason.strip() if isinstance(reason, str) else None,
        )
    return overrides


def _name_exclusion(name: str) -> str | None:
    words = set(re.findall(r"[a-z]+", name.casefold()))
    matches = sorted(words & EXCLUDED_TRACK_NAME_WORDS)
    return matches[0] if matches else None


def _has_valid_duration_note(track: TrackTick) -> bool:
    return any(note.duration > 0 for note in track.notes)


def _is_piano_candidate(track: TrackTick) -> bool:
    return (
        not track.is_drum
        and track.program in PIANO_PROGRAMS
        and _name_exclusion(track.name) is None
        and _has_valid_duration_note(track)
    )


def _is_manual_piano_selection(track: TrackTick) -> bool:
    """Allow a human override to supersede only the conservative name heuristic."""
    return (
        not track.is_drum
        and track.program in PIANO_PROGRAMS
        and _has_valid_duration_note(track)
    )


def select_piano_tracks(
    score: ScoreTick, override: TrackSelectionOverride | None = None
) -> SelectionDecision:
    """Select tracks only when the decision is conservative or explicitly overridden."""
    candidates = tuple(
        index for index, track in enumerate(score.tracks) if _is_piano_candidate(track)
    )

    if override is not None and override.selected_track_indices is not None:
        indices = override.selected_track_indices
        invalid_indices = [
            index for index in indices if not 0 <= index < len(score.tracks)
        ]
        if invalid_indices:
            return SelectionDecision(
                selected_indices=(),
                candidate_indices=candidates,
                cc64_indices=(),
                merge=False,
                needs_review=True,
                warnings=(f"override_track_indices_out_of_range:{invalid_indices}",),
            )

        disallowed = [
            index
            for index in indices
            if not _is_manual_piano_selection(score.tracks[index])
        ]
        if disallowed:
            return SelectionDecision(
                selected_indices=(),
                candidate_indices=candidates,
                cc64_indices=(),
                merge=False,
                needs_review=True,
                warnings=(
                    f"override_selected_non_piano_or_invalid_tracks:{disallowed}",
                ),
            )
        invalid_cc64_indices = [
            index
            for index in override.cc64_track_indices
            if not 0 <= index < len(score.tracks)
            or score.tracks[index].is_drum
            or not any(control.number == 64 for control in score.tracks[index].controls)
            or index in indices
        ]
        if invalid_cc64_indices:
            return SelectionDecision(
                selected_indices=(),
                candidate_indices=candidates,
                cc64_indices=(),
                merge=False,
                needs_review=True,
                warnings=(
                    f"override_invalid_or_duplicate_cc64_tracks:{invalid_cc64_indices}",
                ),
            )
        if len(indices) > 1 and not override.merge:
            return SelectionDecision(
                selected_indices=(),
                candidate_indices=candidates,
                cc64_indices=(),
                merge=False,
                needs_review=True,
                warnings=("multiple_selected_tracks_require_merge_true",),
            )
        return SelectionDecision(
            selected_indices=indices,
            candidate_indices=candidates,
            cc64_indices=override.cc64_track_indices,
            merge=override.merge,
            needs_review=False,
        )

    if len(candidates) == 1:
        return SelectionDecision(candidates, candidates, (), False, False)
    if not candidates:
        return SelectionDecision(
            (),
            candidates,
            (),
            False,
            True,
            ("no_unambiguous_piano_track",),
        )
    return SelectionDecision(
        (),
        candidates,
        (),
        False,
        True,
        (f"ambiguous_piano_tracks:{list(candidates)}",),
    )


def _track_details(index: int, track: TrackTick) -> dict[str, object]:
    return {
        "track_index": index,
        "name": track.name,
        "program": int(track.program),
        "is_drum": bool(track.is_drum),
        "note_count": len(track.notes),
        "cc64_event_count": sum(
            1 for control in track.controls if control.number == 64
        ),
    }


def _clean_selected_track(
    source_track: TrackTick, source_index: int
) -> tuple[TrackTick, list[dict[str, object]]]:
    track = source_track.copy()
    removed: list[dict[str, object]] = []
    for note_index, note in enumerate(track.notes):
        if note.duration <= 0:
            removed.append(
                {
                    "track_index": source_index,
                    "track_name": track.name,
                    "note_index": note_index,
                    "time_ticks": int(note.time),
                    "duration_ticks": int(note.duration),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "reason": "zero_or_negative_duration",
                }
            )
    track.notes.filter(lambda note: note.duration > 0, inplace=True)
    return track, removed


def _merge_tracks(tracks: list[TrackTick]) -> TrackTick:
    merged = tracks[0].copy()
    source_names = [track.name or "unnamed" for track in tracks]
    merged.name = "Merged piano: " + " + ".join(source_names)
    for track in tracks[1:]:
        merged.notes.extend(track.notes)
        merged.controls.extend(track.controls)
        merged.pitch_bends.extend(track.pitch_bends)
        merged.pedals.extend(track.pedals)
        merged.lyrics.extend(track.lyrics)
    merged.notes.sort()
    merged.controls.sort()
    merged.pitch_bends.sort()
    merged.pedals.sort()
    merged.lyrics.sort()
    return merged


def _safe_child(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    child = (root / Path(relative_path)).resolve()
    if not child.is_relative_to(root):
        raise ValueError(f"Path escapes configured root: {relative_path}")
    return child


def _write_midi_atomic(score: ScoreTick, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.cleaning-tmp{output_path.suffix}"
    )
    try:
        score.dump_midi(temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def clean_midi_file(
    *,
    raw_record: dict[str, object],
    input_root: Path,
    output_root: Path,
    override: TrackSelectionOverride | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> CleaningRecord:
    """Prepare one MIDI file while keeping the raw source read-only."""
    relative_value = raw_record.get("relative_path")
    if not isinstance(relative_value, str):
        raise ValueError("Raw inventory record is missing relative_path")
    relative_path = _normalized_relative_path(relative_value)
    source_path = _safe_child(input_root, relative_path)
    output_path = _safe_child(output_root, relative_path)
    record = CleaningRecord(
        relative_path=relative_path,
        source_path=_display_path(source_path),
        output_path=_display_path(output_path),
        status="failed",
        dry_run=dry_run,
    )

    try:
        if not source_path.is_file():
            raise FileNotFoundError(f"Raw MIDI file does not exist: {source_path}")
        expected_hash = raw_record.get("sha256")
        if isinstance(expected_hash, str):
            current_hash = sha256_file(source_path)
            if current_hash != expected_hash:
                raise ValueError(
                    f"Raw file hash differs from inventory: {relative_path}"
                )

        if override is not None and override.reject:
            record.status = "rejected"
            record.rejection_reason = override.reason
            return record
        if override is not None and override.skip:
            record.status = "skipped"
            record.warnings.append("skipped_by_track_selection_override")
            return record

        score = ScoreTick.from_file(source_path)
        decision = select_piano_tracks(score, override)
        record.warnings.extend(decision.warnings)
        if decision.needs_review:
            record.status = "needs_review"
            candidate_set = set(decision.candidate_indices)
            record.removed_tracks = [
                _track_details(index, track)
                for index, track in enumerate(score.tracks)
                if index not in candidate_set
            ]
            return record

        selected_set = set(decision.selected_indices)
        record.selected_tracks = [
            _track_details(index, score.tracks[index])
            for index in decision.selected_indices
        ]
        record.cc64_source_tracks = [
            _track_details(index, score.tracks[index])
            for index in decision.cc64_indices
        ]
        record.removed_tracks = [
            _track_details(index, track)
            for index, track in enumerate(score.tracks)
            if index not in selected_set
        ]

        cleaned_tracks: list[TrackTick] = []
        for index in decision.selected_indices:
            cleaned_track, removed_notes = _clean_selected_track(
                score.tracks[index], index
            )
            cleaned_tracks.append(cleaned_track)
            record.invalid_notes_removed.extend(removed_notes)

        if record.invalid_notes_removed:
            record.warnings.append(
                "removed_zero_or_negative_duration_notes:"
                f"{len(record.invalid_notes_removed)}"
            )

        cleaned_score = score.copy()
        cleaned_score.tracks.clear()
        if decision.merge:
            output_track = _merge_tracks(cleaned_tracks)
            record.warnings.append(
                f"explicitly_merged_piano_tracks:{list(decision.selected_indices)}"
            )
        else:
            output_track = cleaned_tracks[0]

        for index in decision.cc64_indices:
            for control in score.tracks[index].controls:
                if control.number == 64:
                    output_track.controls.append(control.copy())
        output_track.controls.sort()
        cleaned_score.tracks.append(output_track)

        record.notes_kept = sum(len(track.notes) for track in cleaned_tracks)
        record.pedal_events_kept = sum(
            1
            for track in cleaned_tracks
            for control in track.controls
            if control.number == 64
        )
        record.pedal_events_kept += sum(
            1
            for index in decision.cc64_indices
            for control in score.tracks[index].controls
            if control.number == 64
        )

        if output_path.exists() and not overwrite:
            record.status = "skipped"
            record.warnings.append("output_exists_use_overwrite")
        elif dry_run:
            cleaned_score.dumps_midi()
            record.warnings.append("dry_run_no_output_written")
            record.status = "processed"
        else:
            _write_midi_atomic(cleaned_score, output_path)
            record.status = "processed"
    except Exception as exc:  # one bad source must not stop the dataset
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"

    return record


def _write_inventory_atomic(path: Path, records: list[CleaningRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def prepare_dataset(
    *,
    raw_inventory_path: Path,
    track_selections_path: Path,
    input_root: Path,
    output_root: Path,
    processed_inventory_path: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[CleaningRecord]:
    """Prepare all inventoried files and optionally write a processed inventory."""
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root.is_relative_to(input_root):
        raise ValueError("Processed output root cannot be inside the raw input root")
    if processed_inventory_path.resolve().is_relative_to(input_root):
        raise ValueError(
            "Processed inventory cannot be written inside the raw input root"
        )

    raw_records = load_raw_inventory(raw_inventory_path)
    overrides = load_track_selections(track_selections_path)
    records: list[CleaningRecord] = []
    for raw_record in raw_records:
        relative_path = raw_record["relative_path"]
        assert isinstance(relative_path, str)
        records.append(
            clean_midi_file(
                raw_record=raw_record,
                input_root=input_root,
                output_root=output_root,
                override=overrides.get(relative_path),
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )

    if not dry_run:
        _write_inventory_atomic(processed_inventory_path, records)
    return records


def status_counts(records: list[CleaningRecord]) -> dict[str, int]:
    """Count records in each required cleaning status."""
    counts = Counter(record.status for record in records)
    return {
        "processed": counts["processed"],
        "rejected": counts["rejected"],
        "needs_review": counts["needs_review"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
    }
