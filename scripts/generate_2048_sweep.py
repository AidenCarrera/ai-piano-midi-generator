"""Generate the requested 12-sample sweep from the 2,048-context model."""

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
class SweepGroup:
    name: str
    temperature: float
    seeds: tuple[int, ...]


SWEEP_GROUPS = (
    SweepGroup("temperature_0p75", 0.75, (3001, 3002, 3003)),
    SweepGroup("temperature_0p85", 0.85, (3101, 3102, 3103)),
    SweepGroup("temperature_0p95", 0.95, (3201, 3202, 3203)),
    SweepGroup("temperature_1p05", 1.05, (3301, 3302, 3303)),
)

TOP_K = 50
TOP_P = 0.95
TOKEN_COUNT = 2048


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/baseline_2048/best.pt"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/baseline_2048_sweep"),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("reports/baseline_2048_sweep.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/baseline_2048_sweep.md"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _report(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, object],
    device: torch.device,
    records: list[dict[str, object]],
    status: str,
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    for group in SWEEP_GROUPS:
        matching = [record for record in records if record["group"] == group.name]
        if matching:
            summaries.append(
                {
                    "group": group.name,
                    "temperature": group.temperature,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "seeds": list(group.seeds),
                    "summary": summarize_generation_group(matching),
                }
            )
    return {
        "status": status,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": str(device),
        "maximum_new_tokens": TOKEN_COUNT,
        "minimum_new_tokens": TOKEN_COUNT,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "requested_sample_count": sum(len(group.seeds) for group in SWEEP_GROUPS),
        "completed_sample_count": len(records),
        "token_array_contents": (
            "BOS, exactly 2,048 model-generated tokens, then any explicitly "
            "reported state-finalization and EOS tokens"
        ),
        "group_summaries": summaries,
        "samples": records,
    }


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    summaries = report["group_summaries"]
    samples = report["samples"]
    if not isinstance(summaries, list) or not isinstance(samples, list):
        raise ValueError("Sweep report collections are invalid")
    lines = [
        "# 2,048-context sampling sweep",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Samples: {report['completed_sample_count']}",
        "- Model-generated tokens per sample: 2,048",
        "- Top-k: 50",
        "- Top-p: 0.95",
        "",
        "## Group comparison",
        "",
        "| Temperature | Valid | Avg duration (s) | Avg notes | Avg CC64 | "
        "Avg decoder repairs | Avg finalization repairs | Avg blocked invalid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in summaries:
        if not isinstance(entry, dict):
            raise ValueError("Sweep group summary is invalid")
        summary = entry["summary"]
        if not isinstance(summary, dict):
            raise ValueError("Sweep group metrics are invalid")
        lines.append(
            f"| {float(entry['temperature']):.2f} | "
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
            "## Samples",
            "",
            "| Temperature | Seed | Duration (s) | Notes | CC64 | "
            "Decoder repairs | MIDI | Tokens |",
            "|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Sweep sample is invalid")
        repairs = sample["decoder_repairs"]
        if not isinstance(repairs, dict):
            raise ValueError("Sweep decoder repairs are invalid")
        lines.append(
            f"| {float(sample['temperature']):.2f} | {int(sample['seed'])} | "
            f"{float(sample['duration_seconds']):.2f} | "
            f"{int(sample['note_count'])} | {int(sample['cc64_event_count'])} | "
            f"{int(repairs['total'])} | `{sample['output_path']}` | "
            f"`{sample['token_array_path']}` |"
        )
    lines.extend(
        [
            "",
            "Decoder interventions, blocked invalid argmax choices, and final "
            "state repairs are recorded per sample in the JSON report.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = select_device(args.device)
    tokenizer = ExpressiveMidiTokenizer()
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    model_settings = raw_checkpoint.get("model_config")
    if not isinstance(model_settings, dict):
        raise ValueError("Checkpoint is missing model_config")
    model_config = TransformerConfig(**model_settings)
    if model_config.context_length != TOKEN_COUNT:
        raise ValueError("Sweep requires a 2,048-token model context")
    model = PianoTransformer(model_config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    records: list[dict[str, object]] = []
    for group in SWEEP_GROUPS:
        directory = args.output_directory / group.name
        for seed in group.seeds:
            midi_path = directory / f"seed_{seed}.mid"
            token_path = directory / f"seed_{seed}.npy"
            existing = [path for path in (midi_path, token_path) if path.exists()]
            if existing and not args.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")
            record = analyze_generation(
                model,
                tokenizer,
                prompt_tokens=(BOS_TOKEN,),
                mode="unconditional",
                seed=seed,
                output_path=midi_path,
                token_output_path=token_path,
                source_file=None,
                maximum_new_tokens=TOKEN_COUNT,
                minimum_new_tokens=TOKEN_COUNT,
                temperature=group.temperature,
                top_k=TOP_K,
                top_p=TOP_P,
            )
            record.update(
                {
                    "group": group.name,
                    "temperature": group.temperature,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                }
            )
            records.append(record)
            write_json_report(
                args.json_report,
                _report(
                    args=args,
                    checkpoint=raw_checkpoint,
                    device=device,
                    records=records,
                    status="in_progress",
                ),
            )
            print(
                f"[{len(records)}/12] temperature={group.temperature:.2f} "
                f"seed={seed} valid={record['midi_parse_valid']}",
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
    print(json.dumps(report["group_summaries"], indent=2))
    return 0 if all(bool(record["midi_parse_valid"]) for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
