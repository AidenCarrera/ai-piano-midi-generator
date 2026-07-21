"""Generate and compare a reproducible baseline sampling sweep."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from ai_piano_midi_generator.evaluation import (
    analyze_generation,
    summarize_generation_group,
    write_json_report,
)
from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import load_checkpoint, select_device


@dataclass(frozen=True, slots=True)
class SweepSetting:
    name: str
    temperature: float
    top_k: int
    top_p: float


SWEEP_SETTINGS = (
    SweepSetting("temp_0p7_k40_p0p90", 0.7, 40, 0.90),
    SweepSetting("temp_0p8_k40_p0p90", 0.8, 40, 0.90),
    SweepSetting("temp_0p9_k50_p0p95", 0.9, 50, 0.95),
    SweepSetting("temp_1p0_k50_p0p95", 1.0, 50, 0.95),
    SweepSetting("temp_1p1_k80_p0p98", 1.1, 80, 0.98),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/baseline/best.pt"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/baseline_sampling_sweep"),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("reports/baseline_sampling_sweep.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/baseline_sampling_sweep.md"),
    )
    parser.add_argument("--starting-seed", type=int, default=2026)
    parser.add_argument("--samples-per-setting", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    summaries = report["configuration_summaries"]
    if not isinstance(summaries, list):
        raise ValueError("configuration_summaries must be a list")
    lines = [
        "# Baseline sampling sweep",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Seeds per setting: {report['starting_seed']}–{report['ending_seed']}",
        f"- Maximum new tokens: {report['maximum_new_tokens']}",
        f"- Total samples: {report['completed_sample_count']}",
        "",
        "| Temperature | Top-k | Top-p | Valid | Avg duration (s) | "
        "Avg notes | Avg CC64 | Avg decoder repairs | "
        "Avg finalization repairs | Avg blocked invalid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in summaries:
        if not isinstance(entry, dict):
            raise ValueError("Configuration summary must be an object")
        summary = entry["summary"]
        if not isinstance(summary, dict):
            raise ValueError("Configuration summary metrics must be an object")
        lines.append(
            f"| {float(entry['temperature']):.1f} | {int(entry['top_k'])} | "
            f"{float(entry['top_p']):.2f} | "
            f"{int(summary['valid_midi_count'])}/{int(summary['sample_count'])} | "
            f"{float(summary['average_duration_seconds']):.2f} | "
            f"{float(summary['average_note_count']):.1f} | "
            f"{float(summary['average_cc64_event_count']):.1f} | "
            f"{float(summary['average_decoder_repair_count']):.2f} | "
            f"{float(summary['average_finalization_repair_count']):.2f} | "
            f"{float(summary['average_blocked_invalid_event_count']):.2f} |"
        )
    lines.extend(
        [
            "",
            "Blocked invalid events count only cases where the unconstrained "
            "argmax token violated a state constraint and was masked.",
            "Finalization repairs close remaining note/pedal state and append EOS "
            "when the model reaches the token limit.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _report(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, object],
    device: torch.device,
    records: list[dict[str, object]],
    status: str,
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    for setting in SWEEP_SETTINGS:
        matching = [
            record for record in records if record["configuration"] == setting.name
        ]
        if not matching:
            continue
        summaries.append(
            {
                "configuration": setting.name,
                "temperature": setting.temperature,
                "top_k": setting.top_k,
                "top_p": setting.top_p,
                "summary": summarize_generation_group(matching),
            }
        )
    return {
        "status": status,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": str(device),
        "starting_seed": args.starting_seed,
        "ending_seed": args.starting_seed + args.samples_per_setting - 1,
        "samples_per_setting": args.samples_per_setting,
        "maximum_new_tokens": args.max_new_tokens,
        "minimum_new_tokens": 0,
        "requested_sample_count": len(SWEEP_SETTINGS) * args.samples_per_setting,
        "completed_sample_count": len(records),
        "configuration_summaries": summaries,
        "samples": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_setting <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Sample count and token limit must be positive")
    device = select_device(args.device)
    tokenizer = ExpressiveMidiTokenizer()
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    model_settings = raw_checkpoint.get("model_config")
    if not isinstance(model_settings, dict):
        raise ValueError("Checkpoint is missing model_config")
    model = PianoTransformer(TransformerConfig(**model_settings)).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    records: list[dict[str, object]] = []
    for setting in SWEEP_SETTINGS:
        setting_directory = args.output_directory / setting.name
        for offset in range(args.samples_per_setting):
            seed = args.starting_seed + offset
            output = setting_directory / f"seed_{seed}.mid"
            if output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists; use --overwrite: {output}"
                )
            record = analyze_generation(
                model,
                tokenizer,
                prompt_tokens=(BOS_TOKEN,),
                mode="unconditional",
                seed=seed,
                output_path=output,
                source_file=None,
                maximum_new_tokens=args.max_new_tokens,
                minimum_new_tokens=0,
                temperature=setting.temperature,
                top_k=setting.top_k,
                top_p=setting.top_p,
            )
            record.update(
                {
                    "configuration": setting.name,
                    "temperature": setting.temperature,
                    "top_k": setting.top_k,
                    "top_p": setting.top_p,
                }
            )
            records.append(record)
            partial = _report(
                args=args,
                checkpoint=raw_checkpoint,
                device=device,
                records=records,
                status="in_progress",
            )
            write_json_report(args.json_report, partial)
            print(
                f"[{len(records)}/{len(SWEEP_SETTINGS) * args.samples_per_setting}] "
                f"{setting.name} seed={seed} valid={record['midi_parse_valid']}",
                flush=True,
            )

    report = _report(
        args=args,
        checkpoint=raw_checkpoint,
        device=device,
        records=records,
        status="complete",
    )
    write_json_report(args.json_report, report)
    _write_markdown(args.markdown_report, report)
    print(json.dumps(report["configuration_summaries"], indent=2))
    return 0 if all(bool(record["midi_parse_valid"]) for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
