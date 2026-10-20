"""Evaluate the held-out test split and analyze baseline MIDI generations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from statistics import fmean

from torch.utils.data import DataLoader

from ai_piano_midi_generator.dataset import (
    TokenChunkDataset,
    collate_token_chunks,
    default_pad_token_id,
)
from ai_piano_midi_generator.evaluation import (
    analyze_generation,
    evaluate_test_model,
    write_json_report,
    write_markdown_report,
)
from ai_piano_midi_generator.sampling import sanitize_prompt_tokens
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import load_model_checkpoint, select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--json-report",
        type=Path,
        default=Path("reports/baseline_2048/evaluation.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/baseline_2048/evaluation.md"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/baseline_2048/evaluation"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser


def _test_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        if row.get("split") == "test":
            rows.append(row)
    if not rows:
        raise ValueError("No held-out test rows found")
    return sorted(rows, key=lambda row: str(row.get("source_file", "")))


def _prompt_candidates(
    rows: list[dict[str, object]],
    tokenizer: ExpressiveMidiTokenizer,
    prompt_length: int,
) -> list[tuple[str, tuple[str, ...]]]:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for row in rows:
        source_file = row.get("source_file")
        source_path = row.get("source_path")
        if not isinstance(source_file, str) or not isinstance(source_path, str):
            raise ValueError("Test manifest row is missing source paths")
        tokens = tokenizer.encode_file(Path(source_path)).tokens
        prompt, _ = sanitize_prompt_tokens(tokens[:prompt_length], tokenizer)
        candidates.append((source_file, prompt))
    candidates.sort(key=lambda item: (-len(item[1]), item[0].casefold()))
    return candidates


def _mean_note_count(records: list[dict[str, object]], mode: str) -> float:
    values = [
        _record_integer(record, "note_count")
        for record in records
        if record["mode"] == mode
    ]
    return fmean(values) if values else 0.0


def _record_integer(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Generation record {key} must be an integer")
    return value


def _aggregate_generation_counts(
    records: list[dict[str, object]], key: str
) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        value = record[key]
        if not isinstance(value, dict):
            raise ValueError(f"Generation record {key} is not an object")
        totals.update({str(name): int(count) for name, count in value.items()})
    return dict(sorted(totals.items()))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.prompt_length <= 0:
        raise ValueError("Batch size and prompt length must be positive")
    device = select_device(args.device)
    tokenizer = ExpressiveMidiTokenizer()
    model, checkpoint = load_model_checkpoint(args.checkpoint, device=device)
    model.eval()

    test_dataset = TokenChunkDataset(
        args.manifest,
        "test",
        tokenizer,
        context_length=model.config.context_length,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(
            collate_token_chunks,
            pad_token_id=default_pad_token_id(tokenizer),
        ),
    )
    test_metrics = evaluate_test_model(
        model,
        loader,
        tokenizer,
        device=device,
        amp=device.type == "cuda",
    )

    rows = _test_rows(args.manifest)
    candidates = _prompt_candidates(rows, tokenizer, args.prompt_length)
    generations: list[dict[str, object]] = []
    settings = {
        "maximum_new_tokens": args.max_new_tokens,
        "minimum_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    for index, seed in enumerate(args.seeds):
        generations.append(
            analyze_generation(
                model,
                tokenizer,
                prompt_tokens=(BOS_TOKEN,),
                mode="unconditional",
                seed=seed,
                output_path=args.output_directory / f"unconditional_seed_{seed}.mid",
                source_file=None,
                **settings,
            )
        )
        source_file, prompt = candidates[index % len(candidates)]
        generations.append(
            analyze_generation(
                model,
                tokenizer,
                prompt_tokens=prompt,
                mode="prompted",
                seed=seed,
                output_path=args.output_directory / f"prompted_seed_{seed}.mid",
                source_file=source_file,
                **settings,
            )
        )

    comparison = {
        "output_count": len(generations),
        "valid_output_count": sum(
            bool(record["midi_parse_valid"]) for record in generations
        ),
        "unconditional_mean_note_count": _mean_note_count(generations, "unconditional"),
        "prompted_mean_note_count": _mean_note_count(generations, "prompted"),
        "outputs_with_cc64": sum(
            _record_integer(record, "cc64_event_count") > 0 for record in generations
        ),
        "blocked_invalid_events": _aggregate_generation_counts(
            generations, "blocked_invalid_events"
        ),
        "final_state_repairs": _aggregate_generation_counts(
            generations, "final_state_repairs"
        ),
    }
    report: dict[str, object] = {
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_best_validation_loss": float(
            checkpoint.get("best_validation_loss", float("nan"))
        ),
        "device": str(device),
        "test_split_only": True,
        "test_performance_count": len(test_dataset.performance_files),
        "test_chunk_count": len(test_dataset),
        "test_metrics": test_metrics,
        "sampling_settings": {**settings, "seeds": args.seeds},
        "blocked_event_definition": (
            "Counts cases where the unconstrained argmax token violated a "
            "generation constraint and was masked."
        ),
        "generations": generations,
        "comparison": comparison,
    }
    write_json_report(args.json_report, report)
    write_markdown_report(args.markdown_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if comparison["valid_output_count"] == len(generations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
