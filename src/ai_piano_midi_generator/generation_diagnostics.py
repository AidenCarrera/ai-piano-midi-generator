"""Diagnostics for time-shift dominated MIDI token generation."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from ai_piano_midi_generator.tokenizer import ExpressiveMidiTokenizer

DEFAULT_ROLLING_WINDOW = 64


def _shift_steps(token: str) -> int | None:
    if not token.startswith("TIME_SHIFT_"):
        return None
    return int(token.removeprefix("TIME_SHIFT_"))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a threshold from no performances")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _as_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _rolling_time_shift(
    tokens: Sequence[str], window_size: int
) -> tuple[float, list[float]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not tokens:
        return 0.0, []
    flags = [int(_shift_steps(token) is not None) for token in tokens]
    effective_window = min(window_size, len(flags))
    current = sum(flags[:effective_window])
    ratios = [current / effective_window]
    for index in range(effective_window, len(flags)):
        current += flags[index] - flags[index - effective_window]
        ratios.append(current / effective_window)
    return max(ratios), ratios


def analyze_token_timing(
    tokens: Sequence[str],
    tokenizer: ExpressiveMidiTokenizer,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    collapse_ratio: float | None = None,
) -> dict[str, object]:
    """Measure timing, silence, and time-shift repetition in one sequence."""
    resolution = tokenizer.config.timing_resolution_seconds
    maximum_shift = tokenizer.config.maximum_time_shift_steps
    shifts: Counter[int] = Counter()
    runs: list[dict[str, int | float]] = []
    current_run_start: int | None = None
    current_run_count = 0
    current_run_steps = 0
    elapsed_steps = 0
    note_on_steps: list[int] = []

    def close_run(end_index: int) -> None:
        nonlocal current_run_start, current_run_count, current_run_steps
        if current_run_start is not None:
            runs.append(
                {
                    "start_token_index": current_run_start,
                    "end_token_index": end_index,
                    "token_count": current_run_count,
                    "elapsed_seconds": current_run_steps * resolution,
                }
            )
        current_run_start = None
        current_run_count = 0
        current_run_steps = 0

    for index, token in enumerate(tokens):
        shift = _shift_steps(token)
        if shift is not None:
            shifts[shift] += 1
            elapsed_steps += shift
            if current_run_start is None:
                current_run_start = index
            current_run_count += 1
            current_run_steps += shift
            continue
        close_run(index - 1)
        if token.startswith("NOTE_ON_"):
            note_on_steps.append(elapsed_steps)
    close_run(len(tokens) - 1)

    total_seconds = elapsed_steps * resolution
    if note_on_steps:
        time_before_first = note_on_steps[0] * resolution
        note_gaps = [
            (right - left) * resolution
            for left, right in zip(note_on_steps, note_on_steps[1:])
        ]
        trailing_silence = (elapsed_steps - note_on_steps[-1]) * resolution
        maximum_silence = max([time_before_first, trailing_silence, *note_gaps])
        maximum_between = max(note_gaps) if note_gaps else None
    else:
        time_before_first = total_seconds
        maximum_silence = total_seconds
        maximum_between = None

    time_shift_count = sum(shifts.values())
    token_count = len(tokens)
    max_shift_count = shifts[maximum_shift]
    longest_by_tokens = max(runs, key=lambda run: int(run["token_count"]), default=None)
    longest_by_time = max(
        runs, key=lambda run: float(run["elapsed_seconds"]), default=None
    )
    rolling_maximum, rolling_ratios = _rolling_time_shift(tokens, rolling_window)
    collapse_index: int | None = None
    if collapse_ratio is not None:
        effective_window = min(rolling_window, token_count)
        for start, ratio in enumerate(rolling_ratios):
            if ratio > collapse_ratio:
                collapse_index = start + effective_window - 1
                break

    return {
        "token_count": token_count,
        "note_on_count": len(note_on_steps),
        "time_shift_token_count": time_shift_count,
        "time_shift_token_ratio": time_shift_count / token_count
        if token_count
        else 0.0,
        "time_shift_distribution": {
            f"TIME_SHIFT_{steps}": shifts[steps] for steps in sorted(shifts)
        },
        "time_shift_run_count": len(runs),
        "time_shift_runs": runs,
        "longest_time_shift_run_tokens": (
            int(longest_by_tokens["token_count"]) if longest_by_tokens else 0
        ),
        "longest_time_shift_run_seconds": (
            float(longest_by_time["elapsed_seconds"]) if longest_by_time else 0.0
        ),
        "longest_time_shift_run": longest_by_tokens,
        "elapsed_token_duration_seconds": total_seconds,
        "time_before_first_note_on_seconds": time_before_first,
        "maximum_time_between_note_ons_seconds": maximum_between,
        "maximum_silence_seconds": maximum_silence,
        "notes_per_second": (
            len(note_on_steps) / total_seconds if total_seconds > 0 else 0.0
        ),
        "maximum_time_shift_token_count": max_shift_count,
        "maximum_time_shift_token_share": (
            max_shift_count / time_shift_count if time_shift_count else 0.0
        ),
        "rolling_window_tokens": rolling_window,
        "maximum_rolling_time_shift_ratio": rolling_maximum,
        "time_shift_dominance_start_token": collapse_index,
    }


def derive_degeneracy_thresholds(
    training_diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fit one-sided outlier thresholds using training performances only."""
    if not training_diagnostics:
        raise ValueError("Training diagnostics cannot be empty")

    def values(name: str) -> list[float]:
        return [_as_float(record[name], name) for record in training_diagnostics]

    return {
        "reference_token_count": _quantile(values("token_count"), 0.50),
        "extremely_low_note_count": _quantile(values("note_on_count"), 0.01),
        "excessive_total_duration_seconds": _quantile(
            values("elapsed_token_duration_seconds"), 0.99
        ),
        "excessive_silence_seconds": _quantile(values("maximum_silence_seconds"), 0.99),
        "excessive_time_shift_token_ratio": _quantile(
            values("time_shift_token_ratio"), 0.99
        ),
        "repetitive_time_shift_run_tokens": _quantile(
            values("longest_time_shift_run_tokens"), 0.99
        ),
        "repetitive_time_shift_run_seconds": _quantile(
            values("longest_time_shift_run_seconds"), 0.99
        ),
        "maximum_time_shift_token_share": _quantile(
            values("maximum_time_shift_token_share"), 0.99
        ),
        "rolling_time_shift_ratio": _quantile(
            values("maximum_rolling_time_shift_ratio"), 0.99
        ),
        "method": {
            "reference_split": "train",
            "lower_tail": "1st percentile for note count",
            "upper_tail": "99th percentile for all timing/repetition metrics",
            "comparison": "strictly outside the fitted percentile",
            "short_sequence_note_count": (
                "scale the 1st-percentile count by observed/reference token count"
            ),
            "performance_count": len(training_diagnostics),
        },
    }


def classify_degeneracy(
    diagnostics: Mapping[str, object], thresholds: Mapping[str, object]
) -> dict[str, object]:
    """Classify a parseable sequence against fitted real-data limits."""
    reasons: list[str] = []
    observed_token_count = _as_float(diagnostics["token_count"], "token_count")
    reference_token_count = _as_float(
        thresholds["reference_token_count"], "reference_token_count"
    )
    low_note_boundary = _as_float(
        thresholds["extremely_low_note_count"], "extremely_low_note_count"
    ) * min(1.0, observed_token_count / reference_token_count)
    comparisons = (
        (
            "elapsed_token_duration_seconds",
            "excessive_total_duration_seconds",
            "above",
            "excessive_total_duration",
        ),
        (
            "maximum_silence_seconds",
            "excessive_silence_seconds",
            "above",
            "excessive_silence",
        ),
        (
            "time_shift_token_ratio",
            "excessive_time_shift_token_ratio",
            "above",
            "excessive_time_shift_token_percentage",
        ),
    )
    if _as_float(diagnostics["note_on_count"], "note_on_count") < low_note_boundary:
        reasons.append("extremely_low_note_count")
    for metric, threshold, direction, reason in comparisons:
        value = _as_float(diagnostics[metric], metric)
        boundary = _as_float(thresholds[threshold], threshold)
        if (direction == "below" and value < boundary) or (
            direction == "above" and value > boundary
        ):
            reasons.append(reason)

    if _as_float(
        diagnostics["longest_time_shift_run_tokens"],
        "longest_time_shift_run_tokens",
    ) > _as_float(
        thresholds["repetitive_time_shift_run_tokens"],
        "repetitive_time_shift_run_tokens",
    ) or _as_float(
        diagnostics["longest_time_shift_run_seconds"],
        "longest_time_shift_run_seconds",
    ) > _as_float(
        thresholds["repetitive_time_shift_run_seconds"],
        "repetitive_time_shift_run_seconds",
    ):
        reasons.append("repetitive_time_shift_runs")
    maximum_shift_dominates = _as_float(
        diagnostics["maximum_time_shift_token_share"],
        "maximum_time_shift_token_share",
    ) > _as_float(
        thresholds["maximum_time_shift_token_share"],
        "maximum_time_shift_token_share",
    )
    if maximum_shift_dominates:
        reasons.append("maximum_time_shift_token_dominance")
    return {
        "is_degenerate": bool(reasons),
        "reasons": reasons,
        "maximum_time_shift_token_dominates": maximum_shift_dominates,
        "effective_extremely_low_note_count_threshold": low_note_boundary,
    }


def summarize_diagnostics(
    diagnostics: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize per-performance diagnostic distributions."""
    records = list(diagnostics)
    if not records:
        raise ValueError("Cannot summarize no diagnostics")
    metrics = (
        "note_on_count",
        "elapsed_token_duration_seconds",
        "notes_per_second",
        "time_shift_token_ratio",
        "longest_time_shift_run_tokens",
        "longest_time_shift_run_seconds",
        "time_before_first_note_on_seconds",
        "maximum_silence_seconds",
        "maximum_time_shift_token_share",
        "maximum_rolling_time_shift_ratio",
    )
    distributions: dict[str, dict[str, float]] = {}
    for metric in metrics:
        data = np.asarray([_as_float(record[metric], metric) for record in records])
        distributions[metric] = {
            "minimum": float(data.min()),
            "p01": float(np.quantile(data, 0.01)),
            "median": float(np.median(data)),
            "p95": float(np.quantile(data, 0.95)),
            "p99": float(np.quantile(data, 0.99)),
            "maximum": float(data.max()),
        }
    shifts: Counter[str] = Counter()
    for record in records:
        distribution = record["time_shift_distribution"]
        if not isinstance(distribution, Mapping):
            raise ValueError("time_shift_distribution must be a mapping")
        shifts.update({str(name): int(count) for name, count in distribution.items()})
    return {
        "performance_count": len(records),
        "metric_distributions": distributions,
        "time_shift_token_distribution": dict(sorted(shifts.items())),
    }
