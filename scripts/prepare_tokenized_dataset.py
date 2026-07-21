"""CLI for grouped splitting and full-performance token-array preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ai_piano_midi_generator.data.preparation import prepare_tokenized_dataset
from ai_piano_midi_generator.tokenizer import TokenizerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group related tunes, assign leakage-free dataset splits, write one "
            "compact token array per performance, and analyze state-safe chunks."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/midi"),
        help="Root containing cleaned MIDI files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tokens"),
        help="Root for one NumPy token array per complete performance.",
    )
    parser.add_argument(
        "--group-overrides",
        type=Path,
        default=Path("data/manifests/tune_group_overrides.jsonl"),
        help="Versionable manual tune-group override JSONL.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/tokenized_dataset.jsonl"),
        help="Generated tokenized dataset manifest.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/tokenized_dataset_preparation.json"),
        help="Machine-readable split and chunk analysis report.",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/tokenized_dataset_preparation.md"),
        help="Human-readable split and chunk analysis report.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Fixed split seed.")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Context lengths to evaluate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing token arrays if their content differs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = prepare_tokenized_dataset(
            input_root=args.input,
            output_root=args.output,
            group_overrides_path=args.group_overrides,
            dataset_manifest_path=args.manifest,
            report_json_path=args.report_json,
            report_markdown_path=args.report_markdown,
            config=TokenizerConfig(),
            seed=args.seed,
            context_lengths=tuple(args.context_lengths),
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Files tokenized: {report['file_count']}")
    print(f"Tune groups: {report['tune_group_count']}")
    print(f"Total tokens: {report['total_token_count']}")
    print(f"Leakage-free: {report['group_leakage_free']}")
    split_balance = report["split_balance"]
    assert isinstance(split_balance, dict)
    for split in ("train", "validation", "test"):
        item = split_balance[split]
        assert isinstance(item, dict)
        print(
            f"{split}: files={item['file_count']}, groups={item['tune_group_count']}, "
            f"tokens={item['token_count']} ({item['token_percent']}%)"
        )
    context_analysis = report["context_analysis"]
    assert isinstance(context_analysis, dict)
    for context, item in context_analysis.items():
        assert isinstance(item, dict)
        print(
            f"context {context}: chunks={item['chunk_count']}, "
            f"safe starts={item['safe_noninitial_chunk_percent']}%, "
            f"state carry={item['state_reconstruction_chunk_count']}"
        )
    print(f"Potential grouping reviews: {len(report['potential_grouping_reviews'])}")
    return 0 if report["group_leakage_free"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
