from __future__ import annotations

import json
from pathlib import Path

from symusic import ControlChange, Note, Score, Tempo, Track

from ai_piano_midi_generator.tokenization_analysis import analyze_tokenization
from ai_piano_midi_generator.tokenizer import TokenizerConfig


def test_analyze_tokenization_writes_round_trip_reports(tmp_path: Path) -> None:
    input_root = tmp_path / "midi"
    input_root.mkdir()
    score = Score(480)
    score.tempos.append(Tempo(0, 120.0))
    track = Track("Piano", 0, False)
    track.notes.append(Note(0, 480, 60, 90))
    track.controls.extend([ControlChange(0, 64, 127), ControlChange(480, 64, 0)])
    score.tracks.append(track)
    score.dump_midi(input_root / "example.mid")
    json_path = tmp_path / "reports" / "analysis.json"
    markdown_path = tmp_path / "reports" / "analysis.md"

    report = analyze_tokenization(
        input_root=input_root,
        report_json_path=json_path,
        report_markdown_path=markdown_path,
        config=TokenizerConfig(),
    )

    assert report["discovered_file_count"] == 1
    assert report["analyzed_file_count"] == 1
    assert report["failed_file_count"] == 0
    assert report["issues"] == []
    assert report["round_trip"] == {
        "expected_semantic_event_count": 4,
        "decoded_semantic_event_count": 4,
        "semantic_event_retention_percent": 100.0,
        "event_count_match_file_count": 1,
        "event_count_match_file_percent": 100.0,
        "parseable_decoded_file_count": 1,
        "parseable_decoded_file_percent": 100.0,
    }
    assert json.loads(json_path.read_text(encoding="utf-8"))["analyzed_file_count"] == 1
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Expressive MIDI tokenization analysis\n"
    )
