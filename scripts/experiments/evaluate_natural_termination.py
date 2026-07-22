"""Evaluate natural EOS termination for unconditional and prompted generation."""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Literal, Sequence

import numpy as np
import torch

from ai_piano_midi_generator.evaluation import analyze_generation, write_json_report
from ai_piano_midi_generator.generation_diagnostics import (
    analyze_token_timing,
    classify_degeneracy,
)
from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.sampling import sanitize_prompt_tokens
from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    ExpressiveMidiTokenizer,
)
from ai_piano_midi_generator.training import load_checkpoint, select_device

GenerationMode = Literal["unconditional", "prompted"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_natural_termination_2048.toml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/baseline_2048/best.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/tokenized_dataset.jsonl"),
    )
    parser.add_argument(
        "--diagnostic-report",
        type=Path,
        default=Path("reports/baseline_2048/time_shift_diagnostics.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/baseline_2048/natural_termination"),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("reports/baseline_2048/natural_termination.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/baseline_2048/natural_termination.md"),
    )
    parser.add_argument(
        "--audition-report",
        type=Path,
        default=Path("reports/baseline_2048/natural_termination_audition.md"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _config(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    generation = raw.get("generation")
    prompting = raw.get("prompting")
    if not isinstance(generation, dict) or not isinstance(prompting, dict):
        raise ValueError("Config requires [generation] and [prompting]")
    if generation.get("minimum_new_tokens") != 0:
        raise ValueError("Natural termination requires minimum_new_tokens = 0")
    if generation.get("maximum_new_tokens") != 2048:
        raise ValueError("Natural termination study requires a 2,048-token cap")
    return generation, prompting


def _integer(mapping: dict[str, object], name: str) -> int:
    value = mapping.get(name)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(mapping: dict[str, object], name: str) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _number_list(mapping: dict[str, object], name: str) -> list[float]:
    value = mapping.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    if not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} values must be numeric")
    return [float(item) for item in value]


def _integer_list(mapping: dict[str, object], name: str) -> list[int]:
    value = mapping.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"{name} values must be integers")
    return list(value)


def _test_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        if value.get("split") == "test":
            rows.append(value)
    if not rows:
        raise ValueError("Token manifest contains no held-out test files")
    return sorted(rows, key=lambda row: str(row.get("source_file", "")).casefold())


def _selected_prompts(
    rows: list[dict[str, object]], count: int, selection_seed: int
) -> list[dict[str, object]]:
    if len(rows) < count:
        raise ValueError("Not enough held-out files for distinct prompts")
    randomizer = random.Random(selection_seed)
    indices = randomizer.sample(range(len(rows)), count)
    return [rows[index] for index in indices]


def _prompt_tokens(
    row: dict[str, object],
    tokenizer: ExpressiveMidiTokenizer,
    prompt_token_count: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    token_path = row.get("token_array_path")
    if not isinstance(token_path, str):
        raise ValueError("Prompt row is missing token_array_path")
    tokens = tokenizer.ids_to_tokens(np.load(Path(token_path), allow_pickle=False))
    prefix = tuple(token for token in tokens[:prompt_token_count] if token != EOS_TOKEN)
    if not prefix or prefix[0] != BOS_TOKEN:
        prefix = (BOS_TOKEN, *prefix)
    return sanitize_prompt_tokens(prefix, tokenizer)


def _temperature_name(temperature: float) -> str:
    return f"temperature_{temperature:.2f}".replace(".", "p")


def _key(mode: str, temperature: float, seed: int) -> str:
    return f"{mode}|{temperature:.2f}|{seed}"


def _load_thresholds(path: Path) -> tuple[dict[str, object], int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Diagnostic report must be an object")
    thresholds = raw.get("thresholds")
    rolling_window = raw.get("rolling_window_tokens")
    if not isinstance(thresholds, dict) or not isinstance(rolling_window, int):
        raise ValueError("Diagnostic report is missing thresholds")
    return thresholds, rolling_window


def _add_token_diagnostics(
    record: dict[str, object],
    tokenizer: ExpressiveMidiTokenizer,
    thresholds: dict[str, object],
    rolling_window: int,
) -> None:
    token_path = record.get("token_array_path")
    prompt_count = record.get("prompt_token_count")
    generated_count = record.get("model_generated_token_count")
    rolling_limit = thresholds.get("rolling_time_shift_ratio")
    if not isinstance(token_path, str):
        raise ValueError("Generation record is missing token_array_path")
    if not isinstance(prompt_count, int) or not isinstance(generated_count, int):
        raise ValueError("Generation record is missing token counts")
    if not isinstance(rolling_limit, (int, float)):
        raise ValueError("Rolling threshold must be numeric")
    all_tokens = tokenizer.ids_to_tokens(np.load(Path(token_path), allow_pickle=False))
    generated = all_tokens[prompt_count : prompt_count + generated_count]
    diagnostics = analyze_token_timing(
        generated,
        tokenizer,
        rolling_window=rolling_window,
        collapse_ratio=float(rolling_limit),
    )
    record.update(
        {
            "generated_note_on_count": diagnostics["note_on_count"],
            "generated_elapsed_seconds": diagnostics["elapsed_token_duration_seconds"],
            "generated_notes_per_second": diagnostics["notes_per_second"],
            "time_shift_token_percentage": (
                100.0 * _number(diagnostics, "time_shift_token_ratio")
            ),
            "longest_time_shift_run_tokens": diagnostics[
                "longest_time_shift_run_tokens"
            ],
            "longest_time_shift_run_seconds": diagnostics[
                "longest_time_shift_run_seconds"
            ],
            "maximum_silence_seconds": diagnostics["maximum_silence_seconds"],
            "degeneracy_classification": classify_degeneracy(diagnostics, thresholds),
        }
    )


def _nested_total(record: dict[str, object], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return sum(int(count) for count in value.values())


def _nested_integer(record: dict[str, object], name: str, key: str) -> int:
    value = record.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return _integer(value, key)


def _is_degenerate(record: dict[str, object]) -> bool:
    value = record.get("degeneracy_classification")
    if not isinstance(value, dict):
        raise ValueError("degeneracy_classification must be an object")
    result = value.get("is_degenerate")
    if not isinstance(result, bool):
        raise ValueError("is_degenerate must be Boolean")
    return result


def _group_summary(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("Cannot summarize an empty group")
    eos_positions = [
        value
        for record in records
        if isinstance((value := record.get("model_eos_token_position")), int)
    ]
    terminations = Counter(str(record["termination_reason"]) for record in records)
    return {
        "sample_count": len(records),
        "valid_midi_count": sum(bool(record["midi_parse_valid"]) for record in records),
        "natural_eos_count": terminations["model_eos"],
        "maximum_token_cap_count": terminations["maximum_new_tokens"],
        "termination_counts": dict(sorted(terminations.items())),
        "minimum_eos_position": min(eos_positions) if eos_positions else None,
        "median_eos_position": median(eos_positions) if eos_positions else None,
        "maximum_eos_position": max(eos_positions) if eos_positions else None,
        "average_model_generated_tokens": fmean(
            _integer(record, "model_generated_token_count") for record in records
        ),
        "average_generated_note_count": fmean(
            _integer(record, "generated_note_on_count") for record in records
        ),
        "average_total_midi_note_count": fmean(
            _integer(record, "note_count") for record in records
        ),
        "average_midi_duration_seconds": fmean(
            _number(record, "duration_seconds") for record in records
        ),
        "degenerate_count": sum(_is_degenerate(record) for record in records),
        "decoder_repair_count": sum(
            _nested_integer(record, "decoder_repairs", "total") for record in records
        ),
        "finalization_repair_count": sum(
            _nested_total(record, "final_state_repairs") for record in records
        ),
        "blocked_invalid_prediction_count": sum(
            _nested_total(record, "blocked_invalid_events") for record in records
        ),
    }


def _report(
    *,
    args: argparse.Namespace,
    generation: dict[str, object],
    prompting: dict[str, object],
    prompt_rows: list[dict[str, object]],
    records: list[dict[str, object]],
    status: str,
) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for mode in ("unconditional", "prompted"):
        for temperature in _number_list(generation, "temperatures"):
            matching = [
                record
                for record in records
                if record.get("generation_type") == mode
                and record.get("temperature") == temperature
            ]
            if matching:
                groups.append(
                    {
                        "generation_type": mode,
                        "temperature": temperature,
                        "summary": _group_summary(matching),
                    }
                )
    natural = [
        record for record in records if record.get("termination_reason") == "model_eos"
    ]
    capped = [
        record
        for record in records
        if record.get("termination_reason") == "maximum_new_tokens"
    ]
    closure_count = sum(
        _nested_total(record, "final_state_repairs")
        > _nested_integer(record, "final_state_repairs", "appended_eos")
        for record in capped
    )
    return {
        "status": status,
        "checkpoint": args.checkpoint.as_posix(),
        "config": args.config.as_posix(),
        "generation_settings": generation,
        "prompt_settings": prompting,
        "safeguards_enabled": False,
        "selected_prompt_sources": [row.get("source_file") for row in prompt_rows],
        "completed_sample_count": len(records),
        "overall_summary": {
            "valid_midi_count": sum(
                bool(record.get("midi_parse_valid")) for record in records
            ),
            "natural_eos_count": len(natural),
            "nondegenerate_natural_eos_count": sum(
                not _is_degenerate(record) for record in natural
            ),
            "maximum_token_cap_count": len(capped),
            "cap_outputs_with_forced_note_or_pedal_closure": closure_count,
            "degenerate_count": sum(_is_degenerate(record) for record in records),
            "safeguard_intervention_count": sum(
                len(record.get("safeguard_interventions", []))
                for record in records
                if isinstance(record.get("safeguard_interventions"), list)
            ),
            "human_audition_status": "pending",
        },
        "group_summaries": groups,
        "samples": records,
    }


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    groups = report["group_summaries"]
    samples = report["samples"]
    if not isinstance(groups, list) or not isinstance(samples, list):
        raise ValueError("Natural-termination report collections are invalid")
    overall = report.get("overall_summary")
    if not isinstance(overall, dict):
        raise ValueError("Natural-termination overall summary is invalid")
    lines = [
        "# Natural termination evaluation",
        "",
        "EOS was allowed from the first generated token. There was no minimum "
        "generation length and no duration or silence safeguard.",
        "",
        "## Automated findings",
        "",
        f"- Valid MIDI: {_integer(overall, 'valid_midi_count')}/{len(samples)}",
        f"- Natural EOS: {_integer(overall, 'natural_eos_count')}/{len(samples)}",
        "- Nondegenerate natural EOS: "
        f"{_integer(overall, 'nondegenerate_natural_eos_count')}",
        f"- Hit the 2,048-token cap: {_integer(overall, 'maximum_token_cap_count')}",
        "- Cap endings requiring forced note/pedal closure: "
        f"{_integer(overall, 'cap_outputs_with_forced_note_or_pedal_closure')}",
        f"- Dataset-threshold degeneracies: {_integer(overall, 'degenerate_count')}",
        "- Human DAW audition: pending; see the separate audition queue.",
        "",
        "## Group summary",
        "",
        "| Type | Temp | Samples | Natural EOS | Hit cap | Median EOS | "
        "Avg tokens | Avg generated notes | Avg MIDI duration | Degenerate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("summary"), dict):
            raise ValueError("Natural-termination group is invalid")
        summary = group["summary"]
        assert isinstance(summary, dict)
        median_eos = summary["median_eos_position"]
        lines.append(
            f"| {group['generation_type']} | {_number(group, 'temperature'):.2f} | "
            f"{_integer(summary, 'sample_count')} | "
            f"{_integer(summary, 'natural_eos_count')} | "
            f"{_integer(summary, 'maximum_token_cap_count')} | "
            f"{median_eos if median_eos is not None else '-'} | "
            f"{_number(summary, 'average_model_generated_tokens'):.1f} | "
            f"{_number(summary, 'average_generated_note_count'):.1f} | "
            f"{_number(summary, 'average_midi_duration_seconds'):.2f} | "
            f"{_integer(summary, 'degenerate_count')} |"
        )
    lines.extend(
        [
            "",
            "## Samples",
            "",
            "| Type | Temp | Seed | Prompt source | EOS position | Termination | "
            "Generated notes | Total notes | MIDI duration | Decoder repairs | "
            "Final repairs | Degenerate |",
            "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Natural-termination sample is invalid")
        classification = sample["degeneracy_classification"]
        if not isinstance(classification, dict):
            raise ValueError("Degeneracy classification is invalid")
        reasons = classification["reasons"]
        if not isinstance(reasons, list):
            raise ValueError("Degeneracy reasons must be a list")
        eos = sample["model_eos_token_position"]
        lines.append(
            f"| {sample['generation_type']} | {_number(sample, 'temperature'):.2f} | "
            f"{_integer(sample, 'seed')} | {sample.get('prompt_source_file') or '-'} | "
            f"{eos if eos is not None else '-'} | {sample['termination_reason']} | "
            f"{_integer(sample, 'generated_note_on_count')} | "
            f"{_integer(sample, 'note_count')} | "
            f"{_number(sample, 'duration_seconds'):.2f} | "
            f"{_nested_integer(sample, 'decoder_repairs', 'total')} | "
            f"{_nested_total(sample, 'final_state_repairs')} | "
            f"{', '.join(str(reason) for reason in reasons) or 'none'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audition_candidates(records: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for mode in ("unconditional", "prompted"):
        temperatures = sorted(
            {
                _number(record, "temperature")
                for record in records
                if record["generation_type"] == mode
            }
        )
        for temperature in temperatures:
            matching = [
                record
                for record in records
                if record["generation_type"] == mode
                and record["temperature"] == temperature
            ]
            natural = sorted(
                (
                    record
                    for record in matching
                    if record.get("termination_reason") == "model_eos"
                ),
                key=lambda record: _integer(record, "model_generated_token_count"),
            )
            degenerate = sorted(
                (record for record in matching if _is_degenerate(record)),
                key=lambda record: _number(record, "maximum_silence_seconds"),
                reverse=True,
            )
            stable = [record for record in matching if not _is_degenerate(record)]
            stable_durations = sorted(
                _number(record, "duration_seconds") for record in stable
            )
            typical_duration = median(stable_durations) if stable_durations else 0.0
            typical = sorted(
                stable,
                key=lambda record: abs(
                    _number(record, "duration_seconds") - typical_duration
                ),
            )
            choices = (
                (natural[0] if natural else matching[0], "natural_eos_or_shortest"),
                (
                    degenerate[0] if degenerate else matching[0],
                    "degenerate_stress_case",
                ),
                (typical[0] if typical else matching[0], "typical_nondegenerate"),
            )
            seen: set[int] = set()
            for choice, reason in choices:
                seed = _integer(choice, "seed")
                if seed in seen:
                    continue
                seen.add(seed)
                candidate = dict(choice)
                candidate["selection_reason"] = reason
                selected.append(candidate)
    return selected


def _write_audition_report(path: Path, candidates: list[dict[str, object]]) -> None:
    lines = [
        "# Natural-termination DAW audition queue",
        "",
        "Automated metrics cannot determine musical completeness, phrasing, or "
        "harmonic coherence. Audition these length-stratified samples and fill "
        "in the listening columns without changing the generated files.",
        "",
        "| Type | Temp | Seed | Selection | Tokens | Notes | Duration | MIDI | "
        "Too short? | Complete? | Abrupt? | Repetitive? | Harmonic coherence | "
        "Better prompted? | Notes |",
        "|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    for sample in candidates:
        lines.append(
            f"| {sample['generation_type']} | {_number(sample, 'temperature'):.2f} | "
            f"{_integer(sample, 'seed')} | {sample['selection_reason']} | "
            f"{_integer(sample, 'model_generated_token_count')} | "
            f"{_integer(sample, 'note_count')} | "
            f"{_number(sample, 'duration_seconds'):.2f} | "
            f"`{sample['output_path']}` |  |  |  |  |  |  |  |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("Choose either --resume or --overwrite")
    generation, prompting = _config(args.config)
    temperatures = _number_list(generation, "temperatures")
    seeds = _integer_list(generation, "seeds")
    if len(seeds) < 10:
        raise ValueError("At least ten seeds per configuration are required")
    maximum_new_tokens = _integer(generation, "maximum_new_tokens")
    top_k = _integer(generation, "top_k")
    top_p = _number(generation, "top_p")
    prompt_token_count = _integer(prompting, "prompt_token_count")
    selection_seed = _integer(prompting, "selection_seed")

    tokenizer = ExpressiveMidiTokenizer()
    prompt_rows = _selected_prompts(
        _test_rows(args.manifest), len(seeds), selection_seed
    )
    thresholds, rolling_window = _load_thresholds(args.diagnostic_report)
    records: list[dict[str, object]] = []
    if args.resume and args.json_report.exists():
        previous = json.loads(args.json_report.read_text(encoding="utf-8"))
        previous_records = (
            previous.get("samples") if isinstance(previous, dict) else None
        )
        if not isinstance(previous_records, list):
            raise ValueError("Resume report has no sample records")
        records = [record for record in previous_records if isinstance(record, dict)]
    elif not args.overwrite:
        for path in (args.json_report, args.markdown_report, args.audition_report):
            if path.exists():
                raise FileExistsError(
                    f"Report exists; use --resume/--overwrite: {path}"
                )
    completed = {
        _key(
            str(record["generation_type"]),
            _number(record, "temperature"),
            _integer(record, "seed"),
        )
        for record in records
    }

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    model_settings = checkpoint.get("model_config")
    if not isinstance(model_settings, dict):
        raise ValueError("Checkpoint is missing model_config")
    model = PianoTransformer(TransformerConfig(**model_settings)).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    total = 2 * len(temperatures) * len(seeds)
    for mode in ("unconditional", "prompted"):
        generation_type: GenerationMode = mode
        for temperature in temperatures:
            for prompt_index, seed in enumerate(seeds):
                sample_key = _key(mode, temperature, seed)
                if sample_key in completed:
                    continue
                prompt_row = prompt_rows[prompt_index]
                if mode == "prompted":
                    prompt_tokens, prompt_repairs = _prompt_tokens(
                        prompt_row, tokenizer, prompt_token_count
                    )
                    prompt_source = prompt_row.get("source_file")
                else:
                    prompt_tokens = (BOS_TOKEN,)
                    prompt_source = None
                    prompt_repairs = {}
                directory = (
                    args.output_directory / mode / _temperature_name(temperature)
                )
                midi_path = directory / f"seed_{seed}.mid"
                token_path = directory / f"seed_{seed}.npy"
                if not args.overwrite:
                    for path in (midi_path, token_path):
                        if path.exists():
                            raise FileExistsError(
                                f"Output exists; use --resume/--overwrite: {path}"
                            )
                record = analyze_generation(
                    model,
                    tokenizer,
                    prompt_tokens=prompt_tokens,
                    mode=generation_type,
                    seed=seed,
                    output_path=midi_path,
                    token_output_path=token_path,
                    source_file=(
                        str(prompt_source) if prompt_source is not None else None
                    ),
                    maximum_new_tokens=maximum_new_tokens,
                    minimum_new_tokens=0,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generation_mode="musical",
                    maximum_duration_seconds=None,
                    maximum_silence_seconds=None,
                )
                record.update(
                    {
                        "generation_type": mode,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "prompt_source_file": prompt_source,
                        "prompt_state_repairs": prompt_repairs,
                    }
                )
                _add_token_diagnostics(record, tokenizer, thresholds, rolling_window)
                records.append(record)
                write_json_report(
                    args.json_report,
                    _report(
                        args=args,
                        generation=generation,
                        prompting=prompting,
                        prompt_rows=prompt_rows,
                        records=records,
                        status="in_progress",
                    ),
                )
                print(
                    f"[{len(records)}/{total}] {mode} temperature={temperature:.2f} "
                    f"seed={seed} termination={record['termination_reason']} "
                    f"tokens={record['model_generated_token_count']}",
                    flush=True,
                )
    report = _report(
        args=args,
        generation=generation,
        prompting=prompting,
        prompt_rows=prompt_rows,
        records=records,
        status="complete",
    )
    write_json_report(args.json_report, report)
    _write_markdown(args.markdown_report, report)
    _write_audition_report(args.audition_report, _audition_candidates(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
