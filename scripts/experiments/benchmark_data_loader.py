"""Benchmark token-chunk loading across Windows DataLoader worker counts."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai_piano_midi_generator.dataset import (
    TokenChunkDataset,
    collate_token_chunks,
    default_pad_token_id,
)
from ai_piano_midi_generator.tokenizer import ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/tokenized_dataset.jsonl"),
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--measured-epochs", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _consume_epoch(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[int, int]:
    batch_count = 0
    sample_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_count += 1
        sample_count += inputs.shape[0]
    _synchronize(device)
    return batch_count, sample_count


def _benchmark_workers(
    dataset: TokenChunkDataset,
    *,
    workers: int,
    batch_size: int,
    context_length: int,
    warmup_epochs: int,
    measured_epochs: int,
    device: torch.device,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=workers > 0,
        collate_fn=partial(
            collate_token_chunks,
            pad_token_id=default_pad_token_id(dataset.tokenizer),
            padded_length=context_length,
        ),
    )
    for _ in range(warmup_epochs):
        _consume_epoch(loader, device)

    durations: list[float] = []
    batch_count = 0
    sample_count = 0
    for _ in range(measured_epochs):
        _synchronize(device)
        started = time.perf_counter()
        batch_count, sample_count = _consume_epoch(loader, device)
        durations.append(time.perf_counter() - started)

    median_seconds = statistics.median(durations)
    return {
        "workers": workers,
        "persistent_workers": workers > 0,
        "batches_per_epoch": batch_count,
        "samples_per_epoch": sample_count,
        "epoch_seconds": durations,
        "median_epoch_seconds": median_seconds,
        "median_batches_per_second": batch_count / median_seconds,
        "median_samples_per_second": sample_count / median_seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.context_length <= 0 or args.batch_size <= 0:
        raise ValueError("context length and batch size must be positive")
    if args.warmup_epochs < 0 or args.measured_epochs <= 0:
        raise ValueError("warmup epochs must be non-negative and measured positive")
    if any(workers < 0 for workers in args.workers):
        raise ValueError("worker counts must be non-negative")

    device = select_device(args.device)
    tokenizer = ExpressiveMidiTokenizer()
    setup_started = time.perf_counter()
    dataset = TokenChunkDataset(
        args.manifest,
        "train",
        tokenizer,
        context_length=args.context_length,
    )
    dataset_setup_seconds = time.perf_counter() - setup_started
    cached_token_bytes = dataset.cached_token_bytes
    results = [
        _benchmark_workers(
            dataset,
            workers=workers,
            batch_size=args.batch_size,
            context_length=args.context_length,
            warmup_epochs=args.warmup_epochs,
            measured_epochs=args.measured_epochs,
            device=device,
        )
        for workers in args.workers
    ]
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "settings": {
            "manifest": args.manifest.as_posix(),
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs,
            "measured_epochs": args.measured_epochs,
        },
        "dataset": {
            "chunks": len(dataset),
            "setup_seconds": dataset_setup_seconds,
            "cached_token_bytes": cached_token_bytes,
        },
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
