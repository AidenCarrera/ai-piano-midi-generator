"""Compare baseline and EOS model reports with seed-matched generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-training-report", type=Path, required=True)
    parser.add_argument("--candidate-training-report", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--markdown-report", type=Path, required=True)
    return parser


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _load(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _number(mapping: dict[str, Any], name: str) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _integer(mapping: dict[str, Any], name: str) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    value = report.get("samples")
    if not isinstance(value, list) or not value:
        raise ValueError("Natural-termination report has no samples")
    return [_object(sample, "sample") for sample in value]


def _sample_key(sample: dict[str, Any]) -> tuple[str, float, int]:
    mode = sample.get("generation_type")
    if not isinstance(mode, str):
        raise ValueError("Sample generation_type must be a string")
    return mode, _number(sample, "temperature"), _integer(sample, "seed")


def _is_degenerate(sample: dict[str, Any]) -> bool:
    classification = _object(
        sample.get("degeneracy_classification"), "degeneracy_classification"
    )
    value = classification.get("is_degenerate")
    if not isinstance(value, bool):
        raise ValueError("is_degenerate must be Boolean")
    return value


def _metric(
    baseline: dict[str, Any], candidate: dict[str, Any], name: str
) -> dict[str, float | int]:
    baseline_value = _number(baseline, name)
    candidate_value = _number(candidate, name)
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": candidate_value - baseline_value,
    }


def _group_map(report: dict[str, Any]) -> dict[tuple[str, float], dict[str, Any]]:
    groups = report.get("group_summaries")
    if not isinstance(groups, list):
        raise ValueError("group_summaries must be an array")
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for group_value in groups:
        group = _object(group_value, "group")
        mode = group.get("generation_type")
        if not isinstance(mode, str):
            raise ValueError("Group generation_type must be a string")
        key = mode, _number(group, "temperature")
        result[key] = _object(group.get("summary"), "group summary")
    return result


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = _object(report.get("overall_metrics"), "overall_metrics")
    training = _object(report.get("training_metrics"), "training_metrics")
    transitions = _object(report.get("paired_transitions"), "paired_transitions")
    groups = report.get("group_metrics")
    if not isinstance(groups, list):
        raise ValueError("group_metrics must be an array")
    lines = [
        "# Baseline and EOS model comparison",
        "",
        "Both checkpoints were evaluated with identical generation settings, "
        "held-out prompts, temperatures, and seeds.",
        "",
        "## Overall natural termination",
        "",
        "| Metric | Baseline | EOS | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, name in (
        ("Natural EOS", "natural_eos_count"),
        ("Nondegenerate natural EOS", "nondegenerate_natural_eos_count"),
        ("Hit 2,048-token cap", "maximum_token_cap_count"),
        ("Degenerate", "degenerate_count"),
        ("Valid MIDI", "valid_midi_count"),
    ):
        metric = _object(overall.get(name), name)
        lines.append(
            f"| {label} | {_number(metric, 'baseline'):g} | "
            f"{_number(metric, 'candidate'):g} | {_number(metric, 'delta'):+g} |"
        )
    lines.extend(
        [
            "",
            "## Paired seed outcomes",
            "",
            "| Transition | Samples |",
            "|---|---:|",
        ]
    )
    for name, count in sorted(transitions.items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## By generation type and temperature",
            "",
            "| Type | Temp | Old EOS | New EOS | EOS delta | "
            "Old degenerate | New degenerate | Degenerate delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group_value in groups:
        group = _object(group_value, "group metric")
        eos = _object(group.get("natural_eos_count"), "natural_eos_count")
        degenerate = _object(group.get("degenerate_count"), "degenerate_count")
        lines.append(
            f"| {group['generation_type']} | "
            f"{_number(group, 'temperature'):.2f} | "
            f"{_number(eos, 'baseline'):g} | {_number(eos, 'candidate'):g} | "
            f"{_number(eos, 'delta'):+g} | "
            f"{_number(degenerate, 'baseline'):g} | "
            f"{_number(degenerate, 'candidate'):g} | "
            f"{_number(degenerate, 'delta'):+g} |"
        )
    lines.extend(
        [
            "",
            "## Training",
            "",
            "| Metric | Baseline | EOS | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, name in (
        ("Best validation loss", "best_validation_loss"),
        ("Final validation loss", "final_validation_loss"),
        ("Final training loss", "final_training_loss"),
        ("Average tokens/second", "average_tokens_per_second"),
    ):
        metric = _object(training.get(name), name)
        lines.append(
            f"| {label} | {_number(metric, 'baseline'):.6g} | "
            f"{_number(metric, 'candidate'):.6g} | "
            f"{_number(metric, 'delta'):+.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = _load(args.baseline_report)
    candidate = _load(args.candidate_report)
    baseline_training = _load(args.baseline_training_report)
    candidate_training = _load(args.candidate_training_report)
    for name in ("generation_settings", "prompt_settings", "selected_prompt_sources"):
        if baseline.get(name) != candidate.get(name):
            raise ValueError(f"Reports use different {name}")

    baseline_samples = {_sample_key(sample): sample for sample in _samples(baseline)}
    candidate_samples = {_sample_key(sample): sample for sample in _samples(candidate)}
    if baseline_samples.keys() != candidate_samples.keys():
        raise ValueError("Reports do not contain the same mode/temperature/seed keys")

    transitions: Counter[str] = Counter()
    for key, baseline_sample in baseline_samples.items():
        candidate_sample = candidate_samples[key]
        old_termination = str(baseline_sample.get("termination_reason"))
        new_termination = str(candidate_sample.get("termination_reason"))
        transitions[f"termination: {old_termination} -> {new_termination}"] += 1
        old_degenerate = _is_degenerate(baseline_sample)
        new_degenerate = _is_degenerate(candidate_sample)
        transitions[
            f"degenerate: {str(old_degenerate).lower()} -> "
            f"{str(new_degenerate).lower()}"
        ] += 1

    baseline_overall = _object(baseline.get("overall_summary"), "overall_summary")
    candidate_overall = _object(candidate.get("overall_summary"), "overall_summary")
    overall_metrics = {
        name: _metric(baseline_overall, candidate_overall, name)
        for name in (
            "natural_eos_count",
            "nondegenerate_natural_eos_count",
            "maximum_token_cap_count",
            "degenerate_count",
            "valid_midi_count",
        )
    }
    baseline_groups = _group_map(baseline)
    candidate_groups = _group_map(candidate)
    if baseline_groups.keys() != candidate_groups.keys():
        raise ValueError("Reports do not contain the same evaluation groups")
    group_metrics = [
        {
            "generation_type": mode,
            "temperature": temperature,
            "natural_eos_count": _metric(
                baseline_groups[(mode, temperature)],
                candidate_groups[(mode, temperature)],
                "natural_eos_count",
            ),
            "degenerate_count": _metric(
                baseline_groups[(mode, temperature)],
                candidate_groups[(mode, temperature)],
                "degenerate_count",
            ),
        }
        for mode, temperature in sorted(baseline_groups)
    ]
    training_metrics = {
        name: _metric(baseline_training, candidate_training, name)
        for name in (
            "best_validation_loss",
            "final_validation_loss",
            "final_training_loss",
            "average_tokens_per_second",
        )
    }
    report = {
        "baseline_report": args.baseline_report.as_posix(),
        "candidate_report": args.candidate_report.as_posix(),
        "seed_matched_sample_count": len(baseline_samples),
        "generation_settings_match": True,
        "prompt_settings_match": True,
        "selected_prompt_sources_match": True,
        "overall_metrics": overall_metrics,
        "paired_transitions": dict(sorted(transitions.items())),
        "group_metrics": group_metrics,
        "training_metrics": training_metrics,
    }
    _write_json(args.json_report, report)
    _write_markdown(args.markdown_report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
