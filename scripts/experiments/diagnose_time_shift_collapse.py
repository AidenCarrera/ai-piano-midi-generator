"""Compare generated time-shift behavior with the grouped token dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ai_piano_midi_generator.evaluation import write_json_report
from ai_piano_midi_generator.generation_diagnostics import (
    DEFAULT_ROLLING_WINDOW,
    analyze_token_timing,
    classify_degeneracy,
    derive_degeneracy_thresholds,
    summarize_diagnostics,
)
from ai_piano_midi_generator.model import PianoTransformer
from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    PEDAL_UP_TOKEN,
    ExpressiveMidiTokenizer,
)
from ai_piano_midi_generator.training import load_model_checkpoint, select_device

COMPARISON_TOKEN_COUNT = 2048


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/tokenized_dataset.jsonl"),
    )
    parser.add_argument(
        "--sweep-report",
        type=Path,
        default=Path("reports/archive/baseline_2048_forced_sweep.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/baseline_2048/best.pt"),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("reports/baseline_2048/time_shift_diagnostics.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/baseline_2048/time_shift_diagnostics.md"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    return parser


def _manifest_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _musical_tokens(
    token_ids: np.ndarray, tokenizer: ExpressiveMidiTokenizer
) -> tuple[str, ...]:
    tokens = tokenizer.ids_to_tokens(token_ids)
    start = 1 if tokens and tokens[0] == BOS_TOKEN else 0
    end = len(tokens) - 1 if tokens and tokens[-1] == EOS_TOKEN else len(tokens)
    return tokens[start:end]


def _dataset_statistics(
    rows: list[dict[str, object]],
    tokenizer: ExpressiveMidiTokenizer,
    rolling_window: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    full_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    windows_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        split = row.get("split")
        token_path = row.get("token_array_path")
        if not isinstance(split, str) or not isinstance(token_path, str):
            raise ValueError("Manifest row is missing split or token_array_path")
        token_ids = np.load(Path(token_path), allow_pickle=False)
        tokens = _musical_tokens(token_ids, tokenizer)
        full_by_split[split].append(
            analyze_token_timing(tokens, tokenizer, rolling_window=rolling_window)
        )
        full_window_count = len(tokens) // COMPARISON_TOKEN_COUNT
        for window_index in range(full_window_count):
            start = window_index * COMPARISON_TOKEN_COUNT
            window = tokens[start : start + COMPARISON_TOKEN_COUNT]
            diagnostics = analyze_token_timing(
                window, tokenizer, rolling_window=rolling_window
            )
            diagnostics["source_file"] = row.get("source_file")
            diagnostics["window_index"] = window_index
            windows_by_split[split].append(diagnostics)

    required = {"train", "validation", "test"}
    if set(full_by_split) != required or set(windows_by_split) != required:
        raise ValueError("Manifest must provide train, validation, and test data")
    report: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        report[split] = {
            "full_performances": summarize_diagnostics(full_by_split[split]),
            "nonoverlapping_2048_token_windows": summarize_diagnostics(
                windows_by_split[split]
            ),
        }
    return report, windows_by_split["train"]


def _advance_state(
    token: str,
    active_notes: dict[int, int],
    state: dict[str, bool],
) -> None:
    if token.startswith("VELOCITY_"):
        state["velocity"] = True
    elif token.startswith("NOTE_ON_"):
        pitch = int(token.removeprefix("NOTE_ON_"))
        active_notes[pitch] = active_notes.get(pitch, 0) + 1
    elif token.startswith("NOTE_OFF_"):
        pitch = int(token.removeprefix("NOTE_OFF_"))
        if active_notes.get(pitch, 0) > 0:
            active_notes[pitch] -= 1
    elif token == PEDAL_DOWN_TOKEN:
        state["pedal"] = True
    elif token == PEDAL_UP_TOKEN:
        state["pedal"] = False


def _grammar_mask(
    tokenizer: ExpressiveMidiTokenizer,
    active_notes: dict[int, int],
    state: dict[str, bool],
    device: torch.device,
) -> torch.Tensor:
    allowed = torch.ones(tokenizer.vocabulary_size, dtype=torch.bool, device=device)
    allowed[tokenizer.token_to_id[BOS_TOKEN]] = False
    allowed[tokenizer.token_to_id[EOS_TOKEN]] = False
    if not state["velocity"]:
        for pitch in range(128):
            allowed[tokenizer.token_to_id[f"NOTE_ON_{pitch}"]] = False
    for pitch in range(128):
        if active_notes.get(pitch, 0) <= 0:
            allowed[tokenizer.token_to_id[f"NOTE_OFF_{pitch}"]] = False
    allowed[tokenizer.token_to_id[PEDAL_DOWN_TOKEN]] = not state["pedal"]
    allowed[tokenizer.token_to_id[PEDAL_UP_TOKEN]] = state["pedal"]
    return allowed


def _sampling_probabilities(
    logits: torch.Tensor,
    allowed: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    filtered = (logits / temperature).masked_fill(~allowed, -torch.inf)
    threshold = torch.topk(filtered, min(top_k, filtered.numel())).values[-1]
    filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    remove = torch.cumsum(sorted_probabilities, dim=-1) > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    filtered = torch.full_like(filtered, -torch.inf).scatter(
        0, sorted_indices, sorted_logits
    )
    return torch.softmax(filtered, dim=-1)


def _top_tokens(
    probabilities: torch.Tensor,
    tokenizer: ExpressiveMidiTokenizer,
    count: int = 5,
) -> list[dict[str, object]]:
    values, indices = torch.topk(probabilities, count)
    return [
        {"token": tokenizer.vocabulary[int(index)], "probability": float(value)}
        for value, index in zip(values.cpu(), indices.cpu(), strict=True)
    ]


@torch.inference_mode()
def _probability_snapshots(
    model: PianoTransformer,
    tokenizer: ExpressiveMidiTokenizer,
    model_tokens: tuple[str, ...],
    indices: set[int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    prefix = [BOS_TOKEN]
    active_notes: dict[int, int] = {}
    state = {"pedal": False, "velocity": False}
    time_shift_ids = torch.tensor(
        [
            token_id
            for token_id, token in enumerate(tokenizer.vocabulary)
            if token.startswith("TIME_SHIFT_")
        ],
        device=device,
    )
    max_shift_id = tokenizer.token_to_id[
        f"TIME_SHIFT_{tokenizer.config.maximum_time_shift_steps}"
    ]
    results: list[dict[str, object]] = []
    for index, selected_token in enumerate(model_tokens):
        if index in indices:
            input_ids = torch.tensor(
                [tokenizer.token_to_id[token] for token in prefix],
                dtype=torch.long,
                device=device,
            )[-model.config.context_length :]
            logits, _ = model(input_ids.unsqueeze(0))
            next_logits = logits[0, -1].float()
            raw_probabilities = torch.softmax(next_logits, dim=-1)
            allowed = _grammar_mask(tokenizer, active_notes, state, device)
            sampled_probabilities = _sampling_probabilities(
                next_logits,
                allowed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            selected_id = tokenizer.token_to_id[selected_token]
            results.append(
                {
                    "model_token_index": index,
                    "selected_token": selected_token,
                    "selected_logit": float(next_logits[selected_id]),
                    "selected_raw_probability": float(raw_probabilities[selected_id]),
                    "selected_sampling_probability": float(
                        sampled_probabilities[selected_id]
                    ),
                    "raw_time_shift_probability_mass": float(
                        raw_probabilities[time_shift_ids].sum()
                    ),
                    "sampling_time_shift_probability_mass": float(
                        sampled_probabilities[time_shift_ids].sum()
                    ),
                    "raw_maximum_shift_probability": float(
                        raw_probabilities[max_shift_id]
                    ),
                    "sampling_maximum_shift_probability": float(
                        sampled_probabilities[max_shift_id]
                    ),
                    "raw_top_tokens": _top_tokens(raw_probabilities, tokenizer),
                    "sampling_top_tokens": _top_tokens(
                        sampled_probabilities, tokenizer
                    ),
                }
            )
        prefix.append(selected_token)
        _advance_state(selected_token, active_notes, state)
    return results


def _snapshot_indices(diagnostics: dict[str, object], token_count: int) -> set[int]:
    indices: set[int] = set()
    collapse = diagnostics["time_shift_dominance_start_token"]
    if isinstance(collapse, int):
        indices.update(collapse + offset for offset in (-2, -1, 0, 1, 2, 8))
    longest = diagnostics["longest_time_shift_run"]
    if isinstance(longest, dict):
        start = int(longest["start_token_index"])
        end = int(longest["end_token_index"])
        indices.update((start - 1, start, (start + end) // 2, end, end + 1))
    return {index for index in indices if 0 <= index < token_count}


def _generated_statistics(
    sweep: dict[str, object],
    tokenizer: ExpressiveMidiTokenizer,
    thresholds: dict[str, object],
    rolling_window: int,
    model: PianoTransformer,
) -> list[dict[str, object]]:
    samples = sweep.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Sweep report has no samples list")
    records: list[dict[str, object]] = []
    collapse_value = thresholds["rolling_time_shift_ratio"]
    if not isinstance(collapse_value, (int, float)):
        raise ValueError("rolling_time_shift_ratio threshold must be numeric")
    collapse_ratio = float(collapse_value)
    for number, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError("Sweep sample must be an object")
        token_path = sample.get("token_array_path")
        generated_count = sample.get("model_generated_token_count")
        prompt_count = sample.get("prompt_token_count")
        if not isinstance(token_path, str) or not isinstance(generated_count, int):
            raise ValueError("Sweep sample is missing its token array metadata")
        if not isinstance(prompt_count, int):
            raise ValueError("Sweep sample is missing prompt_token_count")
        all_tokens = tokenizer.ids_to_tokens(
            np.load(Path(token_path), allow_pickle=False)
        )
        model_tokens = all_tokens[prompt_count : prompt_count + generated_count]
        diagnostics = analyze_token_timing(
            model_tokens,
            tokenizer,
            rolling_window=rolling_window,
            collapse_ratio=collapse_ratio,
        )
        diagnostics["total_midi_duration_seconds"] = sample.get("duration_seconds")
        diagnostics["classification"] = classify_degeneracy(diagnostics, thresholds)
        snapshots = _probability_snapshots(
            model,
            tokenizer,
            model_tokens,
            _snapshot_indices(diagnostics, len(model_tokens)),
            temperature=float(sample["temperature"]),
            top_k=int(sample["top_k"]),
            top_p=float(sample["top_p"]),
        )
        records.append(
            {
                "temperature": sample["temperature"],
                "seed": sample["seed"],
                "output_path": sample["output_path"],
                "token_array_path": token_path,
                "midi_parse_valid": sample["midi_parse_valid"],
                "cc64_event_count": sample["cc64_event_count"],
                "decoder_repairs": sample["decoder_repairs"],
                "blocked_invalid_events": sample["blocked_invalid_events"],
                "final_state_repairs": sample["final_state_repairs"],
                "diagnostics": diagnostics,
                "probability_snapshots": snapshots,
            }
        )
        print(f"[{number}/{len(samples)}] diagnosed seed {sample['seed']}", flush=True)
    return records


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    thresholds = report["thresholds"]
    samples = report["generated_samples"]
    dataset = report["dataset_statistics"]
    if not isinstance(thresholds, dict) or not isinstance(samples, list):
        raise ValueError("Invalid diagnostic report")
    if not isinstance(dataset, dict):
        raise ValueError("Invalid dataset statistics")
    lines = [
        "# 2,048-context generation degeneracy diagnostics",
        "",
        "The original sweep files and reports were read without modification.",
        "Thresholds use non-overlapping, full 2,048-token windows from the train "
        "split so generated and real sequences have equivalent lengths.",
        "",
        "## Dataset-derived thresholds",
        "",
        "| Classification boundary | Value |",
        "|---|---:|",
    ]
    for name, value in thresholds.items():
        if name != "method":
            lines.append(f"| {name} | {float(value):.6f} |")
    lines.extend(
        [
            "",
            "## Dataset comparison",
            "",
            "| Split | Performances | 2,048-token windows | Median TS % | "
            "99th percentile max silence (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        split_data = dataset[split]
        assert isinstance(split_data, dict)
        full = split_data["full_performances"]
        windows = split_data["nonoverlapping_2048_token_windows"]
        assert isinstance(full, dict) and isinstance(windows, dict)
        metrics = windows["metric_distributions"]
        assert isinstance(metrics, dict)
        ts = metrics["time_shift_token_ratio"]
        silence = metrics["maximum_silence_seconds"]
        assert isinstance(ts, dict) and isinstance(silence, dict)
        lines.append(
            f"| {split} | {int(full['performance_count'])} | "
            f"{int(windows['performance_count'])} | "
            f"{100 * float(ts['median']):.2f}% | {float(silence['p99']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Preserved unconstrained sweep",
            "",
            "| Temp | Seed | MIDI valid | Notes | Duration (s) | Notes/s | TS % | "
            "Longest run | Max silence (s) | Collapse token | Classification |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for sample in samples:
        assert isinstance(sample, dict)
        diagnostics = sample["diagnostics"]
        assert isinstance(diagnostics, dict)
        classification = diagnostics["classification"]
        assert isinstance(classification, dict)
        reasons = classification["reasons"]
        assert isinstance(reasons, list)
        collapse = diagnostics["time_shift_dominance_start_token"]
        lines.append(
            f"| {float(sample['temperature']):.2f} | {int(sample['seed'])} | "
            f"{sample['midi_parse_valid']} | {int(diagnostics['note_on_count'])} | "
            f"{float(diagnostics['total_midi_duration_seconds']):.2f} | "
            f"{float(diagnostics['notes_per_second']):.3f} | "
            f"{100 * float(diagnostics['time_shift_token_ratio']):.2f}% | "
            f"{int(diagnostics['longest_time_shift_run_tokens'])} | "
            f"{float(diagnostics['maximum_silence_seconds']):.2f} | "
            f"{collapse if collapse is not None else '-'} | "
            f"{', '.join(str(reason) for reason in reasons) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "Raw and post-filter probability snapshots around each detected collapse "
            "and longest time-shift run are retained in the JSON report.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rolling_window <= 0:
        raise ValueError("rolling-window must be positive")
    tokenizer = ExpressiveMidiTokenizer()
    rows = _manifest_rows(args.manifest)
    dataset_statistics, train_windows = _dataset_statistics(
        rows, tokenizer, args.rolling_window
    )
    thresholds = derive_degeneracy_thresholds(train_windows)

    device = select_device(args.device)
    model, _ = load_model_checkpoint(args.checkpoint, device=device)
    model.eval()
    sweep = json.loads(args.sweep_report.read_text(encoding="utf-8"))
    if not isinstance(sweep, dict):
        raise ValueError("Sweep report must be an object")
    generated = _generated_statistics(
        sweep, tokenizer, thresholds, args.rolling_window, model
    )
    report: dict[str, object] = {
        "status": "complete",
        "checkpoint": args.checkpoint.as_posix(),
        "source_sweep_report": args.sweep_report.as_posix(),
        "comparison_token_count": COMPARISON_TOKEN_COUNT,
        "rolling_window_tokens": args.rolling_window,
        "thresholds": thresholds,
        "dataset_statistics": dataset_statistics,
        "generated_samples": generated,
    }
    write_json_report(args.json_report, report)
    _write_markdown(args.markdown_report, report)
    classifications = [
        sample["diagnostics"]["classification"]  # type: ignore[index]
        for sample in generated
    ]
    print(
        json.dumps(
            {
                "samples": len(generated),
                "degenerate": sum(
                    bool(classification["is_degenerate"])
                    for classification in classifications
                ),
                "json_report": args.json_report.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
