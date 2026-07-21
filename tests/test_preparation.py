from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from symusic import ControlChange, Note, Score, Tempo, Track

from ai_piano_midi_generator.data.preparation import (
    GroupWeight,
    TuneGroupOverride,
    assign_tune_groups,
    automatic_tune_group,
    deterministic_group_splits,
    plan_state_safe_chunks,
    prepare_tokenized_dataset,
)
from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    PEDAL_UP_TOKEN,
    ExpressiveMidiTokenizer,
    TokenizerConfig,
)


def write_midi(path: Path, pitch: int, *, end_with_pedal_down: bool = False) -> None:
    score = Score(480)
    score.tempos.append(Tempo(0, 120.0))
    track = Track("Piano", 0, False)
    track.notes.append(Note(0, 480, pitch, 80))
    track.controls.append(ControlChange(0, 64, 127))
    if not end_with_pedal_down:
        track.controls.append(ControlChange(480, 64, 0))
    score.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(path)


def test_automatic_and_manual_tune_grouping_is_deterministic() -> None:
    assert automatic_tune_group("BlueBossa1GM.mid") == automatic_tune_group(
        "BlueBossa3GM.mid"
    )
    assert automatic_tune_group("Danny Boy.mid") == automatic_tune_group("Dannyboy.mid")
    assert automatic_tune_group("Yesterday.mid") != automatic_tune_group(
        "Yesterdays1.mid"
    )
    overrides = {"short.mid": TuneGroupOverride("short.mid", "full-tune-name")}

    first = assign_tune_groups(["short.mid", "other.mid"], overrides)
    second = assign_tune_groups(["other.mid", "short.mid"], overrides)

    assert first == second
    assert first["short.mid"] == "full-tune-name"


def test_group_level_splits_are_deterministic_and_leakage_free() -> None:
    groups = [
        GroupWeight(f"group-{index}", 2 if index < 3 else 1, 1000 + index * 37)
        for index in range(20)
    ]

    first = deterministic_group_splits(groups, seed=1234)
    second = deterministic_group_splits(list(reversed(groups)), seed=1234)

    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert len(first) == len(groups)
    performance_splits = [
        first[group.tune_group] for group in groups for _ in range(group.file_count)
    ]
    assert len(performance_splits) == sum(group.file_count for group in groups)


def test_chunk_planning_reconstructs_held_note_and_pedal_state() -> None:
    tokens = (
        BOS_TOKEN,
        PEDAL_DOWN_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_60",
        "TIME_SHIFT_10",
        "TIME_SHIFT_10",
        "TIME_SHIFT_10",
        "TIME_SHIFT_10",
        "NOTE_OFF_60",
        PEDAL_UP_TOKEN,
        EOS_TOKEN,
    )

    plans = plan_state_safe_chunks(tokens, context_length=8)

    assert len(plans) == 2
    assert plans[0].safe_start is True
    assert plans[1].state_reconstruction_required is True
    assert plans[1].active_notes_at_start == 1
    assert plans[1].pedal_down_at_start is True
    assert plans[1].reconstruction_tokens == (
        BOS_TOKEN,
        PEDAL_DOWN_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_60",
    )
    assert all(plan.model_token_count <= 8 for plan in plans)
    assert all(not plan.forced_inside_event_group for plan in plans)


def test_chunk_planning_prefers_safe_boundary_when_available() -> None:
    tokens = (
        BOS_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_60",
        "TIME_SHIFT_10",
        "NOTE_OFF_60",
        "TIME_SHIFT_10",
        "VELOCITY_9",
        "NOTE_ON_64",
        "TIME_SHIFT_10",
        "NOTE_OFF_64",
        EOS_TOKEN,
    )

    plans = plan_state_safe_chunks(tokens, context_length=8)

    assert len(plans) == 2
    assert plans[1].safe_start is True
    assert plans[1].state_reconstruction_required is False
    assert plans[1].reconstruction_tokens == (BOS_TOKEN,)


def test_full_preparation_writes_round_trip_arrays_without_group_leakage(
    tmp_path: Path,
) -> None:
    midi_root = tmp_path / "midi"
    write_midi(midi_root / "Tune Take 1.mid", 60)
    write_midi(midi_root / "Tune Take 2.mid", 62, end_with_pedal_down=True)
    write_midi(midi_root / "Other.mid", 64)
    write_midi(midi_root / "Third.mid", 65)
    overrides = tmp_path / "tune_groups.jsonl"
    overrides.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "relative_path": "Tune Take 1.mid",
                        "tune_group": "tune",
                    }
                ),
                json.dumps(
                    {
                        "relative_path": "Tune Take 2.mid",
                        "tune_group": "tune",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    token_root = tmp_path / "tokens"
    manifest = tmp_path / "tokenized.jsonl"
    report_json = tmp_path / "report.json"
    tokenizer = ExpressiveMidiTokenizer(TokenizerConfig())

    report = prepare_tokenized_dataset(
        input_root=midi_root,
        output_root=token_root,
        group_overrides_path=overrides,
        dataset_manifest_path=manifest,
        report_json_path=report_json,
        report_markdown_path=tmp_path / "report.md",
        config=tokenizer.config,
        seed=7,
        context_lengths=(32,),
    )

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    take_rows = [row for row in rows if row["tune_group"] == "tune"]
    assert len(rows) == 4
    assert {row["split"] for row in take_rows} in (
        {"train"},
        {"validation"},
        {"test"},
    )
    assert report["group_leakage_free"] is True
    assert report["files_ending_with_sustain_down_count"] == 1
    assert report["ending_sustain_sequences_modified"] is False
    assert report["token_array_round_trip_verified"] is True
    for row in rows:
        array = np.load(row["token_array_path"], allow_pickle=False)
        assert array.dtype == np.uint16
        assert array.ndim == 1
        assert len(array) == row["token_count"]
        tokens = tokenizer.ids_to_tokens(array)
        assert tokens[0] == BOS_TOKEN
        assert tokens[-1] == EOS_TOKEN
        assert tokenizer.decode(tokens).repairs.total == 0
        assert row["note_count"] == 1
