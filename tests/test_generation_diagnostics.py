from __future__ import annotations

from ai_piano_midi_generator.generation_diagnostics import (
    analyze_token_timing,
    classify_degeneracy,
    derive_degeneracy_thresholds,
)
from ai_piano_midi_generator.tokenizer import ExpressiveMidiTokenizer


def test_time_shift_run_detection_and_silence_tracking() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    tokens = (
        "TIME_SHIFT_100",
        "TIME_SHIFT_50",
        "VELOCITY_10",
        "NOTE_ON_60",
        "TIME_SHIFT_20",
        "NOTE_ON_64",
        "TIME_SHIFT_30",
    )

    result = analyze_token_timing(tokens, tokenizer, rolling_window=4)

    assert result["time_shift_token_count"] == 4
    assert result["longest_time_shift_run_tokens"] == 2
    assert result["longest_time_shift_run_seconds"] == 1.5
    assert result["time_before_first_note_on_seconds"] == 1.5
    assert result["maximum_time_between_note_ons_seconds"] == 0.2
    assert result["maximum_silence_seconds"] == 1.5
    assert result["elapsed_token_duration_seconds"] == 2.0
    assert result["time_shift_distribution"] == {
        "TIME_SHIFT_20": 1,
        "TIME_SHIFT_30": 1,
        "TIME_SHIFT_50": 1,
        "TIME_SHIFT_100": 1,
    }


def _diagnostic(note_count: float, scale: float) -> dict[str, object]:
    return {
        "token_count": 2048,
        "note_on_count": note_count,
        "elapsed_token_duration_seconds": 10.0 * scale,
        "maximum_silence_seconds": scale,
        "time_shift_token_ratio": 0.1 * scale,
        "longest_time_shift_run_tokens": scale,
        "longest_time_shift_run_seconds": 0.2 * scale,
        "maximum_time_shift_token_share": 0.05 * scale,
        "maximum_rolling_time_shift_ratio": 0.1 * scale,
    }


def test_dataset_derived_thresholds_are_deterministic() -> None:
    records = [_diagnostic(100.0 + index, 1.0 + index / 100) for index in range(100)]

    first = derive_degeneracy_thresholds(records)
    second = derive_degeneracy_thresholds(list(reversed(records)))

    assert first == second
    assert first["extremely_low_note_count"] == 100.99
    method = first["method"]
    assert isinstance(method, dict)
    assert method["reference_split"] == "train"
    assert method["performance_count"] == 100


def test_degeneracy_classification_reports_each_outlier_type() -> None:
    training = [_diagnostic(100.0, 1.0), _diagnostic(120.0, 2.0)]
    thresholds = derive_degeneracy_thresholds(training)
    failed = _diagnostic(1.0, 5.0)

    classification = classify_degeneracy(failed, thresholds)

    assert classification["is_degenerate"] is True
    assert classification["reasons"] == [
        "extremely_low_note_count",
        "excessive_total_duration",
        "excessive_silence",
        "excessive_time_shift_token_percentage",
        "repetitive_time_shift_runs",
        "maximum_time_shift_token_dominance",
    ]


def test_collapse_point_is_first_window_outside_training_limit() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    tokens = tuple(["NOTE_ON_60"] * 4 + ["TIME_SHIFT_100"] * 5)

    result = analyze_token_timing(
        tokens,
        tokenizer,
        rolling_window=4,
        collapse_ratio=0.75,
    )

    assert result["time_shift_dominance_start_token"] == 7
