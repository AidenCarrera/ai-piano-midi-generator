from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from symusic import ControlChange, Note, Score, Tempo, TimeSignature, Track

from ai_piano_midi_generator.data.inspection import (
    discover_midi_files,
    inspect_dataset,
    inspect_midi_file,
)


def write_midi(
    path: Path,
    *,
    program: int = 0,
    is_drum: bool = False,
    with_note: bool = True,
) -> None:
    score = Score(480)
    score.tempos.append(Tempo(0, 120))
    score.time_signatures.append(TimeSignature(0, 4, 4))
    track = Track("Piano", program, is_drum)
    if with_note:
        track.notes.append(Note(0, 480, 60, 90))
    track.controls.append(ControlChange(240, 64, 127))
    score.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(path)


def test_discover_midi_files_is_recursive_and_case_insensitive(
    tmp_path: Path,
) -> None:
    write_midi(tmp_path / "b.MID")
    write_midi(tmp_path / "nested" / "a.mid")
    write_midi(tmp_path / "nested" / "c.MiD")
    (tmp_path / "ignored.midi").write_bytes(b"not selected")

    paths = discover_midi_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "b.MID",
        "nested/a.mid",
        "nested/c.MiD",
    ]


def test_inspect_midi_file_records_required_metadata(tmp_path: Path) -> None:
    midi_path = tmp_path / "nested" / "piano.mid"
    write_midi(midi_path)

    result = inspect_midi_file(midi_path, tmp_path)

    assert result.filename == "piano.mid"
    assert result.relative_path == "nested/piano.mid"
    assert result.sha256 == hashlib.sha256(midi_path.read_bytes()).hexdigest()
    assert result.parse_success is True
    assert result.error is None
    assert result.classification == "accepted"
    assert result.duration_seconds == pytest.approx(0.5)
    assert result.track_count == 1
    assert result.track_names == ["Piano"]
    assert result.midi_programs == [0]
    assert result.drum_or_percussion_tracks == []
    assert result.track_note_counts[0]["note_count"] == 1
    assert result.note_count == 1
    assert result.pitch_range == {"min": 60, "max": 60}
    assert result.velocity_range == {"min": 90, "max": 90}
    assert result.sustain_pedal_cc64_event_count == 1
    assert result.tempo_event_count == 1
    assert result.time_signature_changes == [
        {"time_seconds": 0.0, "numerator": 4, "denominator": 4}
    ]
    assert result.invalid_or_zero_length_notes == []


@pytest.mark.parametrize(
    ("program", "is_drum", "expected_issue"),
    [(40, False, "non_piano_programs"), (0, True, "drum_or_percussion_tracks")],
)
def test_non_piano_and_drum_files_need_review(
    tmp_path: Path,
    program: int,
    is_drum: bool,
    expected_issue: str,
) -> None:
    midi_path = tmp_path / f"review-{program}-{is_drum}.mid"
    write_midi(midi_path, program=program, is_drum=is_drum)

    result = inspect_midi_file(midi_path, tmp_path)

    if is_drum:
        assert result.classification == "rejected"
        assert "no_valid_pitched_notes" in result.issues
    else:
        assert result.classification == "needs_review"
    assert expected_issue in result.issues


def test_empty_midi_is_rejected(tmp_path: Path) -> None:
    midi_path = tmp_path / "empty.mid"
    write_midi(midi_path, with_note=False)

    result = inspect_midi_file(midi_path, tmp_path)

    assert result.parse_success is True
    assert result.classification == "rejected"
    assert result.issues == ["no_notes"]


def test_zero_length_note_is_recorded_and_needs_review(tmp_path: Path) -> None:
    midi_path = tmp_path / "zero-length.mid"
    score = Score(480)
    track = Track("Piano", 0, False)
    track.notes.append(Note(0, 480, 60, 90))
    track.notes.append(Note(480, 0, 64, 80))
    score.tracks.append(track)
    score.dump_midi(midi_path)

    result = inspect_midi_file(midi_path, tmp_path)

    assert result.classification == "needs_review"
    assert result.issues == ["invalid_or_zero_length_notes"]
    assert len(result.invalid_or_zero_length_notes) == 1
    assert result.invalid_or_zero_length_notes[0]["reasons"] == [
        "zero_or_negative_duration"
    ]


def test_dataset_continues_after_malformed_file_and_writes_reports(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    valid_path = input_dir / "valid.mid"
    malformed_path = input_dir / "broken.MID"
    write_midi(valid_path)
    malformed_path.write_bytes(b"not a MIDI file")
    manifest_path = tmp_path / "manifests" / "inventory.jsonl"
    summary_json_path = tmp_path / "reports" / "summary.json"
    summary_markdown_path = tmp_path / "reports" / "summary.md"

    summary = inspect_dataset(
        input_dir,
        manifest_path,
        summary_json_path,
        summary_markdown_path,
    )

    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["relative_path"] == "broken.MID"
    assert records[0]["parse_success"] is False
    assert records[0]["classification"] == "rejected"
    assert records[1]["classification"] == "accepted"
    assert summary["total_midi_files"] == 2
    assert summary["classification_counts"] == {
        "accepted": 1,
        "needs_review": 0,
        "rejected": 1,
    }
    assert json.loads(summary_json_path.read_text()) == summary
    assert "# Dataset inspection summary" in summary_markdown_path.read_text()


def test_outputs_cannot_be_written_inside_raw_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="inside raw input"):
        inspect_dataset(
            input_dir,
            input_dir / "inventory.jsonl",
            tmp_path / "summary.json",
            tmp_path / "summary.md",
        )
