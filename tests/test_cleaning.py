from __future__ import annotations

import hashlib
import json
from pathlib import Path

from symusic import ControlChange, Note, Score, Tempo, Track
from symusic.core import ScoreTick

from ai_piano_midi_generator.data.cleaning import (
    TrackSelectionOverride,
    clean_midi_file,
    prepare_dataset,
    select_piano_tracks,
)


def make_score(path: Path, tracks: list[Track], *, tempo_qpm: float = 123.0) -> None:
    score = Score(480)
    score.tempos.append(Tempo(120, tempo_qpm))
    score.tracks.extend(tracks)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(path)


def piano_track(
    name: str = "Piano",
    *,
    program: int = 0,
    pitch: int = 60,
    duration: int = 480,
) -> Track:
    track = Track(name, program, False)
    track.notes.append(Note(240, duration, pitch, 87))
    track.controls.append(ControlChange(300, 64, 127))
    track.controls.append(ControlChange(700, 64, 0))
    return track


def drum_track() -> Track:
    track = Track("Drums", 0, True)
    track.notes.append(Note(0, 120, 36, 100))
    return track


def raw_record(path: Path, raw_root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(raw_root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_automatic_piano_selection_removes_percussion_and_preserves_events(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = raw_root / "song.mid"
    make_score(source, [piano_track(), drum_track()])

    result = clean_midi_file(
        raw_record=raw_record(source, raw_root),
        input_root=raw_root,
        output_root=output_root,
    )

    output = output_root / "song.mid"
    original = ScoreTick.from_file(source)
    cleaned = ScoreTick.from_file(output)
    assert result.status == "processed"
    assert [track["track_index"] for track in result.selected_tracks] == [0]
    assert [track["track_index"] for track in result.removed_tracks] == [1]
    assert result.notes_kept == 1
    assert result.pedal_events_kept == 2
    assert len(cleaned.tracks) == 1
    note = cleaned.tracks[0].notes[0]
    assert (note.time, note.duration, note.pitch, note.velocity) == (240, 480, 60, 87)
    assert [
        (control.time, control.number, control.value)
        for control in cleaned.tracks[0].controls
    ] == [
        (300, 64, 127),
        (700, 64, 0),
    ]
    assert [(tempo.time, tempo.qpm) for tempo in cleaned.tempos] == [
        (tempo.time, tempo.qpm) for tempo in original.tempos
    ]


def test_ambiguous_piano_tracks_require_manual_selection(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "ambiguous.mid"
    make_score(source, [piano_track("Piano RH"), piano_track("Piano LH", pitch=48)])
    score = ScoreTick.from_file(source)

    decision = select_piano_tracks(score)

    assert decision.needs_review is True
    assert decision.selected_indices == ()
    assert decision.candidate_indices == (0, 1)
    assert decision.warnings == ("ambiguous_piano_tracks:[0, 1]",)


def test_manual_override_selects_one_of_multiple_piano_tracks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = raw_root / "manual.mid"
    make_score(source, [piano_track("Piano RH"), piano_track("Piano LH", pitch=48)])
    inventory = tmp_path / "raw_inventory.jsonl"
    inventory.write_text(json.dumps(raw_record(source, raw_root)) + "\n")
    selections = tmp_path / "track_selections.jsonl"
    selections.write_text(
        json.dumps({"relative_path": "manual.mid", "selected_track_indices": [1]})
        + "\n"
    )

    records = prepare_dataset(
        raw_inventory_path=inventory,
        track_selections_path=selections,
        input_root=raw_root,
        output_root=output_root,
        processed_inventory_path=tmp_path / "processed_inventory.jsonl",
    )

    result = records[0]
    cleaned = ScoreTick.from_file(output_root / "manual.mid")
    assert result.status == "processed"
    assert [track["track_index"] for track in result.selected_tracks] == [1]
    assert len(cleaned.tracks) == 1
    assert cleaned.tracks[0].notes[0].pitch == 48


def test_multiple_tracks_merge_only_with_explicit_merge_override(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "merge.mid"
    make_score(source, [piano_track("Piano RH"), piano_track("Piano LH", pitch=48)])
    record = raw_record(source, raw_root)

    review = clean_midi_file(
        raw_record=record,
        input_root=raw_root,
        output_root=tmp_path / "review-output",
        override=TrackSelectionOverride("merge.mid", (0, 1)),
    )
    merged = clean_midi_file(
        raw_record=record,
        input_root=raw_root,
        output_root=tmp_path / "merged-output",
        override=TrackSelectionOverride("merge.mid", (0, 1), merge=True),
    )

    cleaned = ScoreTick.from_file(tmp_path / "merged-output" / "merge.mid")
    assert review.status == "needs_review"
    assert merged.status == "processed"
    assert len(cleaned.tracks) == 1
    assert sorted(note.pitch for note in cleaned.tracks[0].notes) == [48, 60]
    assert merged.pedal_events_kept == 4


def test_manual_cc64_carrier_preserves_controls_without_unrelated_notes(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = raw_root / "carrier.mid"
    carrier = Track("Controller carrier", 24, False)
    carrier.notes.append(Note(0, 120, 55, 70))
    carrier.controls.append(ControlChange(100, 64, 127))
    carrier.controls.append(ControlChange(900, 64, 0))
    lower = piano_track("Lower", pitch=48)
    upper = piano_track("Upper", pitch=72)
    lower.controls.clear()
    upper.controls.clear()
    make_score(source, [carrier, lower, upper])

    result = clean_midi_file(
        raw_record=raw_record(source, raw_root),
        input_root=raw_root,
        output_root=output_root,
        override=TrackSelectionOverride(
            "carrier.mid", (1, 2), cc64_track_indices=(0,), merge=True
        ),
    )

    cleaned = ScoreTick.from_file(output_root / "carrier.mid")
    assert result.status == "processed"
    assert sorted(note.pitch for note in cleaned.tracks[0].notes) == [48, 72]
    assert [
        (control.time, control.value) for control in cleaned.tracks[0].controls
    ] == [
        (100, 127),
        (900, 0),
    ]
    assert result.pedal_events_kept == 2
    assert [track["track_index"] for track in result.cc64_source_tracks] == [0]


def test_rejected_override_records_reason_without_writing_output(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "reject.mid"
    make_score(source, [piano_track()])

    result = clean_midi_file(
        raw_record=raw_record(source, raw_root),
        input_root=raw_root,
        output_root=tmp_path / "processed",
        override=TrackSelectionOverride(
            "reject.mid", reject=True, reason="Ambiguous source"
        ),
    )

    assert result.status == "rejected"
    assert result.rejection_reason == "Ambiguous source"
    assert not (tmp_path / "processed").exists()


def test_invalid_duration_notes_are_removed_and_logged(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = raw_root / "invalid.mid"
    track = piano_track()
    track.notes.append(Note(960, 0, 64, 70))
    make_score(source, [track])
    raw_hash_before = hashlib.sha256(source.read_bytes()).hexdigest()

    result = clean_midi_file(
        raw_record=raw_record(source, raw_root),
        input_root=raw_root,
        output_root=output_root,
    )

    cleaned = ScoreTick.from_file(output_root / "invalid.mid")
    assert result.status == "processed"
    assert result.notes_kept == 1
    assert len(result.invalid_notes_removed) == 1
    assert result.invalid_notes_removed[0]["duration_ticks"] == 0
    assert len(cleaned.tracks[0].notes) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == raw_hash_before


def test_dry_run_writes_nothing_and_preserves_raw_file(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "dry.mid"
    make_score(source, [piano_track()])
    raw_hash_before = hashlib.sha256(source.read_bytes()).hexdigest()
    raw_inventory = tmp_path / "raw_inventory.jsonl"
    raw_inventory.write_text(json.dumps(raw_record(source, raw_root)) + "\n")
    selections = tmp_path / "track_selections.jsonl"
    selections.write_text("")
    output_root = tmp_path / "processed"
    processed_inventory = tmp_path / "processed_inventory.jsonl"

    records = prepare_dataset(
        raw_inventory_path=raw_inventory,
        track_selections_path=selections,
        input_root=raw_root,
        output_root=output_root,
        processed_inventory_path=processed_inventory,
        dry_run=True,
    )

    assert records[0].status == "processed"
    assert "dry_run_no_output_written" in records[0].warnings
    assert not output_root.exists()
    assert not processed_inventory.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == raw_hash_before


def test_existing_output_is_skipped_without_overwrite(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = raw_root / "existing.mid"
    output = output_root / "existing.mid"
    make_score(source, [piano_track()])
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing output")

    result = clean_midi_file(
        raw_record=raw_record(source, raw_root),
        input_root=raw_root,
        output_root=output_root,
    )

    assert result.status == "skipped"
    assert output.read_bytes() == b"existing output"


def test_failure_does_not_stop_other_files(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    good = raw_root / "good.mid"
    bad = raw_root / "bad.mid"
    make_score(good, [piano_track()])
    bad.write_bytes(b"not midi")
    inventory = tmp_path / "raw_inventory.jsonl"
    inventory.write_text(
        "\n".join(
            [
                json.dumps(raw_record(bad, raw_root)),
                json.dumps(raw_record(good, raw_root)),
            ]
        )
        + "\n"
    )
    selections = tmp_path / "track_selections.jsonl"
    selections.write_text("")

    records = prepare_dataset(
        raw_inventory_path=inventory,
        track_selections_path=selections,
        input_root=raw_root,
        output_root=tmp_path / "processed",
        processed_inventory_path=tmp_path / "processed_inventory.jsonl",
    )

    assert [record.status for record in records] == ["failed", "processed"]
    assert (tmp_path / "processed" / "good.mid").is_file()
