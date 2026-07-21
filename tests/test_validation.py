from __future__ import annotations

import hashlib
import json
from pathlib import Path

from symusic import ControlChange, Note, Score, Track

from ai_piano_midi_generator.data.cleaning import prepare_dataset
from ai_piano_midi_generator.data.validation import validate_final_dataset


def write_midi(path: Path, *, piano: bool) -> int:
    score = Score(480)
    if piano:
        piano_part = Track("Piano", 0, False)
        piano_part.notes.append(Note(0, 480, 60, 90))
        piano_part.controls.append(ControlChange(100, 64, 127))
        piano_part.controls.append(ControlChange(500, 64, 0))
        score.tracks.append(piano_part)
        drums = Track("Drums", 0, True)
        drums.notes.append(Note(0, 120, 36, 100))
        score.tracks.append(drums)
    else:
        instrument = Track("Vibraphone", 11, False)
        instrument.notes.append(Note(0, 480, 72, 80))
        score.tracks.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(path)
    return sum(len(track.notes) for track in score.tracks)


def inventory_record(path: Path, raw_root: Path, note_count: int) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(raw_root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "note_count": note_count,
    }


def test_final_validation_parses_outputs_and_reports_dataset_statistics(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    usable = raw_root / "usable.mid"
    rejected = raw_root / "rejected.mid"
    usable_notes = write_midi(usable, piano=True)
    rejected_notes = write_midi(rejected, piano=False)
    raw_inventory = tmp_path / "raw_inventory.jsonl"
    raw_inventory.write_text(
        "\n".join(
            [
                json.dumps(inventory_record(usable, raw_root, usable_notes)),
                json.dumps(inventory_record(rejected, raw_root, rejected_notes)),
            ]
        )
        + "\n"
    )
    selections = tmp_path / "track_selections.jsonl"
    selections.write_text(
        json.dumps(
            {
                "relative_path": "rejected.mid",
                "reject": True,
                "reason": "No piano track",
            }
        )
        + "\n"
    )
    processed_inventory = tmp_path / "processed_inventory.jsonl"
    output_root = tmp_path / "processed"

    prepare_dataset(
        raw_inventory_path=raw_inventory,
        track_selections_path=selections,
        input_root=raw_root,
        output_root=output_root,
        processed_inventory_path=processed_inventory,
    )
    report = validate_final_dataset(
        raw_inventory_path=raw_inventory,
        processed_inventory_path=processed_inventory,
        output_root=output_root,
        report_json_path=tmp_path / "validation.json",
        report_markdown_path=tmp_path / "validation.md",
    )

    assert report["validation_passed"] is True
    assert report["run_status_counts"] == {
        "processed": 1,
        "rejected": 1,
        "needs_review": 0,
        "skipped": 0,
        "failed": 0,
    }
    assert report["final_usable_file_count"] == 1
    assert report["raw_note_count"] == usable_notes + rejected_notes
    assert report["processed_note_count"] == 1
    assert report["selected_track_count"] == 1
    assert report["discarded_track_count"] == 1
    assert report["files_with_cc64_count"] == 1
    assert report["remaining_invalid_note_count"] == 0
    assert report["output_filenames_unique"] is True
    assert report["outputs_traceable"] is True
    assert json.loads((tmp_path / "validation.json").read_text()) == report
    assert "Validation passed: **True**" in (tmp_path / "validation.md").read_text()


def test_final_validation_reports_nonblocking_piano_range_outlier(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "extended.mid"
    score = Score(480)
    track = Track("Piano", 0, False)
    track.notes.append(Note(0, 480, 109, 64))
    score.tracks.append(track)
    source.parent.mkdir(parents=True)
    score.dump_midi(source)
    raw_inventory = tmp_path / "raw_inventory.jsonl"
    raw_inventory.write_text(json.dumps(inventory_record(source, raw_root, 1)) + "\n")
    selections = tmp_path / "track_selections.jsonl"
    selections.write_text("")
    processed_inventory = tmp_path / "processed_inventory.jsonl"
    output_root = tmp_path / "processed"

    prepare_dataset(
        raw_inventory_path=raw_inventory,
        track_selections_path=selections,
        input_root=raw_root,
        output_root=output_root,
        processed_inventory_path=processed_inventory,
    )
    report = validate_final_dataset(
        raw_inventory_path=raw_inventory,
        processed_inventory_path=processed_inventory,
        output_root=output_root,
        report_json_path=tmp_path / "validation.json",
        report_markdown_path=tmp_path / "validation.md",
    )

    suspicious = report["suspicious_outputs"]
    assert isinstance(suspicious, dict)
    assert suspicious["pitches_outside_88_key_piano_range"] == [
        {"relative_path": "extended.mid", "count": 1, "min": 109, "max": 109}
    ]
    assert report["out_of_88_key_piano_range_note_count"] == 1
    assert report["validation_passed"] is True
