"""Held-out token metrics and generation analysis for trained checkpoints."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
import numpy as np
from symusic.core import ScoreSecond
from torch.nn import functional as F

from ai_piano_midi_generator.model import PianoTransformer
from ai_piano_midi_generator.sampling import SamplingResult, sample_tokens
from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    PEDAL_UP_TOKEN,
    ExpressiveMidiTokenizer,
)

TokenCategory = Literal[
    "note_on",
    "note_off",
    "time_shift",
    "velocity",
    "pedal",
    "bos",
    "eos",
]
TOKEN_CATEGORIES: tuple[TokenCategory, ...] = (
    "note_on",
    "note_off",
    "time_shift",
    "velocity",
    "pedal",
    "bos",
    "eos",
)


@dataclass(slots=True)
class _PrefixState:
    active_notes: dict[int, int]
    pedal_down: bool = False
    velocity_selected: bool = False


def token_category(token: str) -> TokenCategory:
    """Map every tokenizer token to its evaluation category."""
    if token.startswith("NOTE_ON_"):
        return "note_on"
    if token.startswith("NOTE_OFF_"):
        return "note_off"
    if token.startswith("TIME_SHIFT_"):
        return "time_shift"
    if token.startswith("VELOCITY_"):
        return "velocity"
    if token in (PEDAL_DOWN_TOKEN, PEDAL_UP_TOKEN):
        return "pedal"
    if token == BOS_TOKEN:
        return "bos"
    if token == EOS_TOKEN:
        return "eos"
    raise ValueError(f"Unknown token category: {token}")


def _advance_prefix(state: _PrefixState, token: str) -> None:
    if token.startswith("VELOCITY_"):
        state.velocity_selected = True
    elif token.startswith("NOTE_ON_"):
        pitch = int(token.removeprefix("NOTE_ON_"))
        state.active_notes[pitch] = state.active_notes.get(pitch, 0) + 1
    elif token.startswith("NOTE_OFF_"):
        pitch = int(token.removeprefix("NOTE_OFF_"))
        if state.active_notes.get(pitch, 0) > 0:
            state.active_notes[pitch] -= 1
    elif token == PEDAL_DOWN_TOKEN:
        state.pedal_down = True
    elif token == PEDAL_UP_TOKEN:
        state.pedal_down = False


def invalid_prediction_tendencies(
    input_ids: torch.Tensor,
    predicted_ids: torch.Tensor,
    valid_targets: torch.Tensor,
    tokenizer: ExpressiveMidiTokenizer,
) -> dict[str, int]:
    """Count invalid argmax predictions against each observed prefix state."""
    counts: Counter[str] = Counter()
    for row in range(input_ids.size(0)):
        state = _PrefixState(active_notes=defaultdict(int))
        for column in range(input_ids.size(1)):
            if not bool(valid_targets[row, column]):
                continue
            observed = tokenizer.vocabulary[int(input_ids[row, column])]
            _advance_prefix(state, observed)
            predicted = tokenizer.vocabulary[int(predicted_ids[row, column])]
            if predicted == PEDAL_DOWN_TOKEN and state.pedal_down:
                counts["duplicate_pedal_down"] += 1
            elif predicted == PEDAL_UP_TOKEN and not state.pedal_down:
                counts["pedal_up_while_up"] += 1
            elif predicted.startswith("NOTE_OFF_"):
                pitch = int(predicted.removeprefix("NOTE_OFF_"))
                if state.active_notes.get(pitch, 0) <= 0:
                    counts["orphan_note_off"] += 1
            elif predicted.startswith("NOTE_ON_") and not state.velocity_selected:
                counts["note_on_without_velocity"] += 1
            elif predicted == EOS_TOKEN:
                active_count = sum(state.active_notes.values())
                if active_count or state.pedal_down:
                    counts["unsafe_eos"] += 1
                    counts["unclosed_notes"] += active_count
    for name in (
        "duplicate_pedal_down",
        "pedal_up_while_up",
        "orphan_note_off",
        "note_on_without_velocity",
        "unsafe_eos",
        "unclosed_notes",
    ):
        counts.setdefault(name, 0)
    return dict(sorted(counts.items()))


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def evaluate_test_model(
    model: PianoTransformer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    tokenizer: ExpressiveMidiTokenizer,
    *,
    device: torch.device,
    amp: bool = True,
) -> dict[str, object]:
    """Evaluate next-token metrics and unconstrained state tendencies."""
    model.eval()
    total_loss = 0.0
    evaluated_tokens = 0
    correct_tokens = 0
    category_counts: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    started = time.perf_counter()

    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits, _ = model(inputs)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        predictions = logits.argmax(dim=-1)
        valid = targets != -100
        batch_token_count = int(valid.sum().item())
        total_loss += float(loss_sum.detach().cpu())
        evaluated_tokens += batch_token_count
        correct = predictions.eq(targets) & valid
        correct_tokens += int(correct.sum().item())

        valid_target_ids = targets[valid].detach().cpu().tolist()
        valid_correct = correct[valid].detach().cpu().tolist()
        for token_id, is_correct in zip(valid_target_ids, valid_correct, strict=True):
            category = token_category(tokenizer.vocabulary[int(token_id)])
            category_counts[category] += 1
            category_correct[category] += int(bool(is_correct))

        tendencies = invalid_prediction_tendencies(
            inputs.detach().cpu(),
            predictions.detach().cpu(),
            valid.detach().cpu(),
            tokenizer,
        )
        invalid_counts.update(tendencies)

    if evaluated_tokens == 0:
        raise ValueError("Test loader produced no evaluated tokens")
    elapsed = max(time.perf_counter() - started, 1e-9)
    mean_loss = total_loss / evaluated_tokens
    by_category: dict[str, dict[str, int | float | None]] = {}
    for category in TOKEN_CATEGORIES:
        count = category_counts[category]
        category_value: dict[str, int | float | None] = {
            "correct": category_correct[category],
            "count": count,
            "accuracy": category_correct[category] / count if count else None,
        }
        by_category[category] = category_value

    return {
        "test_loss": mean_loss,
        "test_perplexity": math.exp(min(20.0, mean_loss)),
        "next_token_accuracy": correct_tokens / evaluated_tokens,
        "token_accuracy_by_category": by_category,
        "invalid_state_tendencies": {
            name: invalid_counts[name]
            for name in (
                "duplicate_pedal_down",
                "pedal_up_while_up",
                "orphan_note_off",
                "note_on_without_velocity",
                "unsafe_eos",
                "unclosed_notes",
            )
        },
        "parameter_count": model.parameter_count(),
        "evaluated_token_count": evaluated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": evaluated_tokens / elapsed,
    }


def analyze_generation(
    model: PianoTransformer,
    tokenizer: ExpressiveMidiTokenizer,
    *,
    prompt_tokens: tuple[str, ...],
    mode: Literal["unconditional", "prompted"],
    seed: int,
    output_path: Path,
    token_output_path: Path | None = None,
    source_file: str | None,
    maximum_new_tokens: int,
    minimum_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> dict[str, object]:
    """Sample, decode, write, and independently parse one MIDI output."""
    sampled: SamplingResult = sample_tokens(
        model,
        tokenizer,
        prompt_tokens=prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
        minimum_new_tokens=minimum_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        constrain_states=True,
    )
    decoded = tokenizer.decode(sampled.tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded.score.dump_midi(output_path)
    if token_output_path is not None:
        token_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            token_output_path,
            tokenizer.tokens_to_ids(sampled.tokens),
            allow_pickle=False,
        )
    parse_valid = False
    parse_error: str | None = None
    note_count = 0
    cc64_count = 0
    duration_seconds = 0.0
    try:
        score = ScoreSecond.from_file(output_path)
        note_count = sum(len(track.notes) for track in score.tracks)
        cc64_count = sum(
            1
            for track in score.tracks
            for control in track.controls
            if control.number == 64
        )
        duration_seconds = float(score.end())
        parse_valid = True
    except Exception as exc:  # pragma: no cover - defensive report path
        parse_error = f"{type(exc).__name__}: {exc}"

    repair_token_count = sum(sampled.final_state_repairs.values())
    generated_with_repairs = len(sampled.tokens) - len(prompt_tokens)
    model_generated_tokens = max(0, generated_with_repairs - repair_token_count)
    return {
        "mode": mode,
        "seed": seed,
        "source_file": source_file,
        "output_path": output_path.as_posix(),
        "token_array_path": (
            token_output_path.as_posix() if token_output_path is not None else None
        ),
        "prompt_token_count": len(prompt_tokens),
        "model_generated_token_count": model_generated_tokens,
        "total_output_token_count": len(sampled.tokens),
        "ended_with_safe_model_eos": not bool(
            sampled.final_state_repairs.get("appended_eos", 0)
        ),
        "note_count": note_count,
        "cc64_event_count": cc64_count,
        "duration_seconds": duration_seconds,
        "decoder_repairs": decoded.repairs.to_dict(),
        "blocked_invalid_events": sampled.invalid_events_blocked,
        "final_state_repairs": sampled.final_state_repairs,
        "midi_parse_valid": parse_valid,
        "parse_error": parse_error,
    }


def summarize_generation_group(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate comparable generation statistics for one sampling setting."""
    if not records:
        raise ValueError("Cannot summarize an empty generation group")
    blocked_totals: Counter[str] = Counter()
    decoder_totals: Counter[str] = Counter()
    finalization_totals: Counter[str] = Counter()
    durations: list[float] = []
    notes: list[int] = []
    pedal_events: list[int] = []
    generated_tokens: list[int] = []
    valid_count = 0
    safe_eos_count = 0
    for record in records:
        duration = record["duration_seconds"]
        note_count = record["note_count"]
        cc64_count = record["cc64_event_count"]
        generated_count = record["model_generated_token_count"]
        if not isinstance(duration, (int, float)):
            raise ValueError("Generation duration_seconds must be numeric")
        if not isinstance(note_count, int):
            raise ValueError("Generation note_count must be an integer")
        if not isinstance(cc64_count, int):
            raise ValueError("Generation cc64_event_count must be an integer")
        if not isinstance(generated_count, int):
            raise ValueError(
                "Generation model_generated_token_count must be an integer"
            )
        durations.append(float(duration))
        notes.append(note_count)
        pedal_events.append(cc64_count)
        generated_tokens.append(generated_count)
        valid_count += int(bool(record["midi_parse_valid"]))
        safe_eos_count += int(bool(record["ended_with_safe_model_eos"]))
        for field, destination in (
            ("blocked_invalid_events", blocked_totals),
            ("decoder_repairs", decoder_totals),
            ("final_state_repairs", finalization_totals),
        ):
            value = record[field]
            if not isinstance(value, dict):
                raise ValueError(f"Generation record {field} is not an object")
            destination.update({str(name): int(count) for name, count in value.items()})
    sample_count = len(records)
    decoder_repair_count = decoder_totals["total"]
    finalization_repair_count = sum(finalization_totals.values())
    blocked_count = sum(blocked_totals.values())
    return {
        "sample_count": sample_count,
        "valid_midi_count": valid_count,
        "safe_model_eos_count": safe_eos_count,
        "average_generated_token_count": sum(generated_tokens) / sample_count,
        "average_duration_seconds": sum(durations) / sample_count,
        "average_note_count": sum(notes) / sample_count,
        "average_cc64_event_count": sum(pedal_events) / sample_count,
        "average_decoder_repair_count": decoder_repair_count / sample_count,
        "average_finalization_repair_count": (finalization_repair_count / sample_count),
        "average_blocked_invalid_event_count": blocked_count / sample_count,
        "decoder_repair_totals": dict(sorted(decoder_totals.items())),
        "final_state_repair_totals": dict(sorted(finalization_totals.items())),
        "blocked_invalid_event_totals": dict(sorted(blocked_totals.items())),
    }


def write_json_report(path: Path, report: dict[str, object]) -> None:
    """Atomically write a JSON evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    """Write a concise human-readable evaluation comparison."""
    metrics = report["test_metrics"]
    assert isinstance(metrics, dict)
    categories = metrics["token_accuracy_by_category"]
    assert isinstance(categories, dict)
    tendencies = metrics["invalid_state_tendencies"]
    assert isinstance(tendencies, dict)
    generations = report["generations"]
    assert isinstance(generations, list)
    lines = [
        "# Baseline evaluation",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Test loss: {float(metrics['test_loss']):.6f}",
        f"- Test perplexity: {float(metrics['test_perplexity']):.4f}",
        f"- Next-token accuracy: {float(metrics['next_token_accuracy']):.2%}",
        f"- Evaluated tokens: {int(metrics['evaluated_token_count']):,}",
        f"- Throughput: {float(metrics['tokens_per_second']):,.0f} tokens/s",
        "",
        "## Accuracy by token category",
        "",
        "| Category | Correct | Count | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for category in TOKEN_CATEGORIES:
        value = categories[category]
        assert isinstance(value, dict)
        accuracy = value["accuracy"]
        rendered = "n/a" if accuracy is None else f"{float(accuracy):.2%}"
        lines.append(
            f"| {category} | {int(value['correct']):,} | "
            f"{int(value['count']):,} | {rendered} |"
        )
    lines.extend(
        [
            "",
            "## Invalid-state argmax tendencies",
            "",
            "| Tendency | Count |",
            "|---|---:|",
        ]
    )
    for name, value in tendencies.items():
        lines.append(f"| {name} | {int(value):,} |")
    lines.extend(
        [
            "",
            "## Generation comparison",
            "",
            "| Mode | Seed | New tokens | Notes | CC64 | Duration (s) | "
            "Valid MIDI | Decoder repairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for generation in generations:
        assert isinstance(generation, dict)
        repairs = generation["decoder_repairs"]
        assert isinstance(repairs, dict)
        lines.append(
            f"| {generation['mode']} | {generation['seed']} | "
            f"{generation['model_generated_token_count']} | "
            f"{generation['note_count']} | {generation['cc64_event_count']} | "
            f"{float(generation['duration_seconds']):.2f} | "
            f"{generation['midi_parse_valid']} | {repairs['total']} |"
        )
    comparison = report["comparison"]
    assert isinstance(comparison, dict)
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Valid outputs: {comparison['valid_output_count']}/"
            f"{comparison['output_count']}",
            f"- Unconditional mean notes: "
            f"{float(comparison['unconditional_mean_note_count']):.1f}",
            f"- Prompted mean notes: "
            f"{float(comparison['prompted_mean_note_count']):.1f}",
            f"- Outputs containing CC64: {comparison['outputs_with_cc64']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
