"""Train, smoke-test, or tiny-overfit the decoder-only piano Transformer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import tomllib
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai_piano_midi_generator.dataset import (
    TokenChunkDataset,
    collate_token_chunks,
    default_pad_token_id,
)
from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.tokenizer import ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import (
    TrainingConfig,
    configure_determinism,
    evaluate_loss,
    select_device,
    set_random_seed,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_baseline_2048.toml"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--overfit-chunks", type=int, default=2)
    parser.add_argument("--overfit-steps", type=int, default=120)
    return parser


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Missing TOML table: {name}")
    return value


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _loader(
    dataset: TokenChunkDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=partial(
            collate_token_chunks,
            pad_token_id=default_pad_token_id(dataset.tokenizer),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume is not None and (args.smoke_test or args.tiny_overfit):
        parser.error("--resume cannot be combined with diagnostic training modes")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.overfit_chunks <= 0 or args.overfit_steps <= 0:
        parser.error("tiny-overfit chunk and step counts must be positive")
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    data_config = _mapping(raw.get("data"), "data")
    model_config = TransformerConfig(**_mapping(raw.get("model"), "model"))
    training_config = TrainingConfig(**_mapping(raw.get("training"), "training"))
    output_config = _mapping(raw.get("output"), "output")
    manifest = Path(str(data_config["manifest"]))
    context_length = int(data_config["context_length"])
    if context_length != model_config.context_length:
        raise ValueError("data.context_length and model.context_length must match")
    device = select_device(args.device)
    tokenizer = ExpressiveMidiTokenizer()
    if tokenizer.vocabulary_size != model_config.vocabulary_size:
        raise ValueError("Tokenizer and model vocabulary sizes differ")
    set_random_seed(training_config.seed)
    configure_determinism(training_config.deterministic)

    maximum_train_chunks = None
    maximum_validation_chunks = None
    if args.smoke_test:
        maximum_train_chunks = 1
        maximum_validation_chunks = 1
        training_config = replace(
            training_config,
            batch_size=1,
            gradient_accumulation_steps=1,
            maximum_steps=2,
            warmup_steps=0,
            evaluation_interval=1,
            evaluation_batches=1,
            checkpoint_interval=1,
            log_interval=1,
        )
    elif args.tiny_overfit:
        maximum_train_chunks = args.overfit_chunks
        maximum_validation_chunks = args.overfit_chunks
        training_config = replace(
            training_config,
            batch_size=1,
            gradient_accumulation_steps=1,
            maximum_steps=args.overfit_steps,
            learning_rate=1e-3,
            minimum_learning_rate=1e-4,
            warmup_steps=min(5, max(0, args.overfit_steps - 1)),
            evaluation_interval=max(1, args.overfit_steps // 6),
            evaluation_batches=args.overfit_chunks,
            checkpoint_interval=args.overfit_steps,
            log_interval=max(1, args.overfit_steps // 12),
        )

    train_dataset = TokenChunkDataset(
        manifest,
        "train",
        tokenizer,
        context_length=context_length,
        maximum_chunks=maximum_train_chunks,
    )
    validation_dataset = TokenChunkDataset(
        manifest,
        "validation" if not args.tiny_overfit else "train",
        tokenizer,
        context_length=context_length,
        maximum_chunks=maximum_validation_chunks,
    )
    pin_memory = device.type == "cuda"
    train_loader = _loader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=not args.tiny_overfit and not args.smoke_test,
        workers=training_config.data_loader_workers,
        seed=training_config.seed,
        pin_memory=pin_memory,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        workers=training_config.data_loader_workers,
        seed=training_config.seed,
        pin_memory=pin_memory,
    )
    model = PianoTransformer(model_config).to(device)
    initial_loss = (
        None
        if args.resume is not None
        else evaluate_loss(
            model,
            validation_loader,
            device=device,
            maximum_batches=training_config.evaluation_batches,
            amp=training_config.amp,
        )
    )

    configured_checkpoint_directory = Path(str(output_config["checkpoint_directory"]))
    configured_report_path = Path(str(output_config["training_report"]))
    temporary_checkpoints: TemporaryDirectory[str] | None = None
    if args.smoke_test:
        temporary_checkpoints = TemporaryDirectory(prefix="ai-piano-smoke-")
        checkpoint_directory = Path(temporary_checkpoints.name)
        report_path = configured_report_path.with_name("smoke_test.json")
        step_limit = 1
    elif args.tiny_overfit:
        temporary_checkpoints = TemporaryDirectory(prefix="ai-piano-overfit-")
        checkpoint_directory = Path(temporary_checkpoints.name)
        report_path = configured_report_path.with_name("tiny_overfit.json")
        step_limit = args.overfit_steps
    else:
        checkpoint_directory = configured_checkpoint_directory
        report_path = configured_report_path
        step_limit = (
            training_config.maximum_steps if args.max_steps is None else args.max_steps
        )

    summary = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        training_config=training_config,
        device=device,
        checkpoint_directory=checkpoint_directory,
        resume_path=args.resume,
        maximum_steps=step_limit,
    )
    if temporary_checkpoints is not None:
        temporary_checkpoints.cleanup()
    report: dict[str, object] = {
        **summary,
        "initial_loss": initial_loss,
        "initial_perplexity": (
            math.exp(min(20.0, initial_loss)) if initial_loss is not None else None
        ),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "dataset_manifest": manifest.as_posix(),
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "context_length": context_length,
        "train_chunk_count": len(train_dataset),
        "validation_chunk_count": len(validation_dataset),
        "mode": "smoke"
        if args.smoke_test
        else "tiny_overfit"
        if args.tiny_overfit
        else "full",
    }
    final_loss = float(summary["final_validation_loss"])
    report["loss_reduction_percent"] = (
        round((initial_loss - final_loss) * 100.0 / initial_loss, 6)
        if initial_loss is not None
        else None
    )
    _write_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if (
        args.tiny_overfit
        and isinstance(report["loss_reduction_percent"], float)
        and report["loss_reduction_percent"] < 20.0
    ):
        print("Tiny-overfit loss did not fall by the required 20%.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
