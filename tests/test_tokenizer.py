from __future__ import annotations

from symusic import ControlChange, Note, Score, Tempo, Track
from symusic.core import ScoreSecond, ScoreTick

from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    PEDAL_UP_TOKEN,
    ExpressiveMidiTokenizer,
    TokenizerConfig,
)


def seconds_score(score: ScoreTick) -> ScoreSecond:
    return ScoreSecond.from_midi(score.dumps_midi())


def base_score() -> tuple[ScoreTick, Track]:
    score = Score(480)
    score.tempos.append(Tempo(0, 120.0))
    track = Track("Piano", 0, False)
    score.tracks.append(track)
    return score, track


def decoded_seconds(
    tokenizer: ExpressiveMidiTokenizer, tokens: tuple[str, ...]
) -> tuple[ScoreSecond, int]:
    decoded = tokenizer.decode(tokens)
    reparsed = ScoreSecond.from_midi(decoded.score.dumps_midi())
    return reparsed, decoded.repairs.total


def test_vocabulary_size_and_configurable_ranges() -> None:
    tokenizer = ExpressiveMidiTokenizer(
        TokenizerConfig(
            timing_resolution_seconds=0.02,
            maximum_time_shift_seconds=0.5,
            velocity_bin_count=16,
        )
    )

    assert tokenizer.config.maximum_time_shift_steps == 25
    assert tokenizer.vocabulary_size == 301
    assert tokenizer.vocabulary[0:4] == (
        BOS_TOKEN,
        EOS_TOKEN,
        PEDAL_DOWN_TOKEN,
        PEDAL_UP_TOKEN,
    )
    assert "NOTE_ON_0" in tokenizer.token_to_id
    assert "NOTE_OFF_127" in tokenizer.token_to_id
    assert "VELOCITY_15" in tokenizer.token_to_id
    assert "TIME_SHIFT_25" in tokenizer.token_to_id
    assert all("TEMPO" not in token for token in tokenizer.vocabulary)


def test_round_trip_preserves_chord_velocity_timing_and_sustain() -> None:
    score, track = base_score()
    for pitch in (60, 64, 67):
        track.notes.append(Note(0, 480, pitch, 64))
    track.controls.append(ControlChange(0, 64, 127))
    track.controls.append(ControlChange(480, 64, 0))
    tokenizer = ExpressiveMidiTokenizer()

    encoded = tokenizer.encode(seconds_score(score))
    decoded, repair_count = decoded_seconds(tokenizer, encoded.tokens)

    assert encoded.tokens[:6] == (
        BOS_TOKEN,
        PEDAL_DOWN_TOKEN,
        "VELOCITY_16",
        "NOTE_ON_60",
        "NOTE_ON_64",
        "NOTE_ON_67",
    )
    assert encoded.tokens[-5:] == (
        "NOTE_OFF_60",
        "NOTE_OFF_64",
        "NOTE_OFF_67",
        PEDAL_UP_TOKEN,
        EOS_TOKEN,
    )
    assert repair_count == 0
    assert [
        (note.pitch, note.time, note.duration) for note in decoded.tracks[0].notes
    ] == [
        (60, 0.0, 0.5),
        (64, 0.0, 0.5),
        (67, 0.0, 0.5),
    ]
    assert all(abs(note.velocity - 64) <= 2 for note in decoded.tracks[0].notes)
    assert [
        (control.time, control.value) for control in decoded.tracks[0].controls
    ] == [(0.0, 127), (0.5, 0)]
    assert max(encoded.diagnostics.timing_errors_seconds) < 1e-9
    assert max(encoded.diagnostics.velocity_errors) <= 2


def test_redundant_cc64_values_do_not_emit_pedal_tokens() -> None:
    score, track = base_score()
    track.notes.append(Note(0, 960, 60, 64))
    for time, value in ((0, 0), (0, 100), (240, 127), (480, 30), (720, 0)):
        track.controls.append(ControlChange(time, 64, value))
    tokenizer = ExpressiveMidiTokenizer()

    tokens = tokenizer.encode(seconds_score(score)).tokens
    pedal_tokens = [t for t in tokens if t in (PEDAL_DOWN_TOKEN, PEDAL_UP_TOKEN)]

    assert pedal_tokens == [PEDAL_DOWN_TOKEN, PEDAL_UP_TOKEN]


def test_round_trip_preserves_overlapping_and_repeated_pitches() -> None:
    score, track = base_score()
    track.notes.append(Note(0, 480, 60, 50))
    track.notes.append(Note(240, 480, 60, 100))
    track.notes.append(Note(120, 600, 62, 75))
    tokenizer = ExpressiveMidiTokenizer()

    encoded = tokenizer.encode(seconds_score(score))
    decoded, repair_count = decoded_seconds(tokenizer, encoded.tokens)

    assert encoded.tokens.count("NOTE_ON_60") == 2
    assert encoded.tokens.count("NOTE_OFF_60") == 2
    assert repair_count == 0
    assert len(decoded.tracks[0].notes) == 3
    repeated = [note for note in decoded.tracks[0].notes if note.pitch == 60]
    assert [(note.time, note.duration) for note in repeated] == [
        (0.0, 0.5),
        (0.25, 0.5),
    ]


def test_tempo_changes_become_elapsed_time_without_tempo_tokens() -> None:
    score, track = base_score()
    score.tempos.append(Tempo(480, 60.0))
    track.notes.append(Note(480, 480, 72, 90))
    tokenizer = ExpressiveMidiTokenizer()

    encoded = tokenizer.encode(seconds_score(score))
    decoded, repair_count = decoded_seconds(tokenizer, encoded.tokens)

    assert all("TEMPO" not in token for token in encoded.tokens)
    assert repair_count == 0
    note = decoded.tracks[0].notes[0]
    assert abs(note.time - 0.5) < 1e-9
    assert abs(note.duration - 1.0) < 1e-9


def test_long_elapsed_time_uses_bounded_shift_tokens() -> None:
    score, track = base_score()
    track.notes.append(Note(960, 480, 60, 80))
    tokenizer = ExpressiveMidiTokenizer(
        TokenizerConfig(
            timing_resolution_seconds=0.01,
            maximum_time_shift_seconds=0.25,
        )
    )

    tokens = tokenizer.encode(seconds_score(score)).tokens
    first_note = tokens.index("NOTE_ON_60")
    shifts = [
        int(token.removeprefix("TIME_SHIFT_"))
        for token in tokens[:first_note]
        if token.startswith("TIME_SHIFT_")
    ]

    assert shifts == [25, 25, 25, 25]
    assert max(shifts) <= tokenizer.config.maximum_time_shift_steps


def test_sub_resolution_note_is_extended_to_one_step_before_decoding() -> None:
    score, track = base_score()
    track.notes.append(Note(0, 1, 60, 80))
    tokenizer = ExpressiveMidiTokenizer()

    encoded = tokenizer.encode(seconds_score(score))
    decoded = tokenizer.decode(encoded.tokens)
    reparsed = ScoreTick.from_midi(decoded.score.dumps_midi())

    assert encoded.diagnostics.minimum_duration_repairs == 1
    assert decoded.repairs.total == 0
    assert reparsed.tracks[0].notes[0].duration > 0


def test_decoder_repairs_orphan_and_unclosed_notes_deterministically() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    decoded = tokenizer.decode(["NOTE_OFF_60", "NOTE_ON_60", EOS_TOKEN, "NOTE_ON_61"])
    reparsed = ScoreTick.from_midi(decoded.score.dumps_midi())

    assert decoded.repairs.missing_bos == 1
    assert decoded.repairs.orphan_note_offs == 1
    assert decoded.repairs.note_ons_without_velocity == 1
    assert decoded.repairs.unclosed_notes == 1
    assert decoded.repairs.tokens_after_eos == 1
    assert len(reparsed.tracks[0].notes) == 1
    assert reparsed.tracks[0].notes[0].duration > 0
