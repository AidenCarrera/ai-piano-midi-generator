"""Plain-PyTorch training, evaluation, scheduling, and checkpoint utilities."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    maximum_steps: int = 20_000
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    evaluation_interval: int = 250
    evaluation_batches: int = 40
    checkpoint_interval: int = 500
    log_interval: int = 10
    seed: int = 2026
    data_loader_workers: int = 0
    amp: bool = True
    deterministic: bool = False
    fused_adamw: bool = True

    def __post_init__(self) -> None:
        positive_integers = (
            self.batch_size,
            self.gradient_accumulation_steps,
            self.maximum_steps,
            self.evaluation_interval,
            self.evaluation_batches,
            self.checkpoint_interval,
            self.log_interval,
        )
        if any(value <= 0 for value in positive_integers):
            raise ValueError("Training step and batch settings must be positive")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("Learning rates must be non-negative and peak positive")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if self.warmup_steps < 0 or self.warmup_steps >= self.maximum_steps:
            raise ValueError("warmup_steps must be in [0, maximum_steps)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.data_loader_workers < 0:
            raise ValueError("data_loader_workers must be non-negative")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """Select strict reproducibility or faster backend algorithm selection."""
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=False)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available unless CPU is explicitly requested."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unknown device selection: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learning_rate_at_step(step: int, config: TrainingConfig) -> float:
    """Linear warmup followed by cosine decay."""
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = max(1, config.maximum_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + cosine * (
        config.learning_rate - config.minimum_learning_rate
    )


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Create device-appropriate AdamW with selective weight decay."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and not name.endswith("position_embedding.weight"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    trainable_parameters = [*decay, *no_decay]
    use_fused = config.fused_adamw and bool(trainable_parameters) and all(
        parameter.device.type == "cuda" for parameter in trainable_parameters
    )
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=use_fused,
    )


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def evaluate_loss(
    model: PianoTransformer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    maximum_batches: int,
    amp: bool,
) -> float:
    """Evaluate mean next-token loss over a bounded number of batches."""
    if maximum_batches <= 0:
        raise ValueError("maximum_batches must be positive")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    evaluated_tokens = 0
    try:
        for batch_index, (inputs, targets) in enumerate(batches):
            if batch_index >= maximum_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            token_count = int((targets != -100).sum().item())
            if token_count == 0:
                continue
            with _autocast(device, amp):
                _, loss = model(inputs, targets)
            assert loss is not None
            total_loss += float(loss.detach().cpu()) * token_count
            evaluated_tokens += token_count
    finally:
        model.train(was_training)
    if evaluated_tokens == 0:
        raise ValueError("Evaluation loader produced no batches")
    return total_loss / evaluated_tokens


def save_checkpoint(
    path: Path,
    *,
    model: PianoTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    best_validation_loss: float,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
) -> None:
    """Atomically save model, optimizer, scaler, configuration, and RNG state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_checkpoint(
    path: Path, *, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    """Read a trusted project checkpoint and validate its outer structure."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return checkpoint


def load_checkpoint(
    path: Path,
    *,
    model: PianoTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: torch.device | str = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load model and optional optimizer/scaler state from a checkpoint."""
    checkpoint = _read_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint.get("scaler", {}))
    if restore_rng:
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch_rng_state = checkpoint["torch_rng_state"]
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("Checkpoint torch_rng_state is not a tensor")
        torch.set_rng_state(torch_rng_state.cpu())
        cuda_state = checkpoint.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            if not isinstance(cuda_state, list) or not all(
                isinstance(state, torch.Tensor) for state in cuda_state
            ):
                raise ValueError("Checkpoint cuda_rng_state is invalid")
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return checkpoint


def load_model_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[PianoTransformer, dict[str, Any]]:
    """Load a model and metadata from one trusted project checkpoint read."""
    checkpoint = _read_checkpoint(path, map_location="cpu")
    model_settings = checkpoint.get("model_config")
    if not isinstance(model_settings, dict):
        raise ValueError("Checkpoint is missing model_config")
    model = PianoTransformer(TransformerConfig(**model_settings))
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint is missing model state")
    model.load_state_dict(model_state)
    return model.to(device), checkpoint


def _next_batch(
    iterator: Any, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_model(
    *,
    model: PianoTransformer,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    training_config: TrainingConfig,
    device: torch.device,
    checkpoint_directory: Path,
    resume_path: Path | None = None,
    maximum_steps: int | None = None,
) -> dict[str, float | int | str]:
    """Train, validate, checkpoint, and report baseline language-model metrics."""
    set_random_seed(training_config.seed)
    configure_determinism(training_config.deterministic)
    model.to(device)
    optimizer = build_optimizer(model, training_config)
    amp_enabled = training_config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_step = 0
    best_validation_loss = math.inf
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
            restore_rng=True,
        )
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])

    step_limit = (
        training_config.maximum_steps if maximum_steps is None else maximum_steps
    )
    if step_limit <= 0:
        raise ValueError("maximum_steps must be positive")
    if start_step >= step_limit:
        raise ValueError(
            f"Checkpoint step {start_step} has already reached step limit {step_limit}"
        )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    iterator = iter(train_loader)
    running_loss = 0.0
    running_tokens = 0
    total_tokens = 0
    training_start = time.perf_counter()
    interval_start = time.perf_counter()
    last_loss = math.nan
    last_validation_loss = math.nan
    final_learning_rate = training_config.learning_rate

    for step in range(start_step, step_limit):
        learning_rate = learning_rate_at_step(step, training_config)
        final_learning_rate = learning_rate
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        step_tokens = 0
        for _ in range(training_config.gradient_accumulation_steps):
            (inputs, targets), iterator = _next_batch(iterator, train_loader)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            step_tokens += int((targets != -100).sum().item())
            with _autocast(device, amp_enabled):
                _, loss = model(inputs, targets)
                assert loss is not None
                scaled_loss = loss / training_config.gradient_accumulation_steps
            scaled_loss_tensor = scaler.scale(scaled_loss)
            assert isinstance(scaled_loss_tensor, torch.Tensor)
            scaled_loss_tensor.backward()
            accumulated_loss += float(loss.detach().cpu())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        last_loss = accumulated_loss / training_config.gradient_accumulation_steps
        running_loss += last_loss
        running_tokens += step_tokens
        total_tokens += step_tokens

        completed_step = step + 1
        if completed_step % training_config.log_interval == 0:
            elapsed = max(time.perf_counter() - interval_start, 1e-9)
            metrics = {
                "step": completed_step,
                "training_loss": running_loss / training_config.log_interval,
                "training_perplexity": math.exp(
                    min(20.0, running_loss / training_config.log_interval)
                ),
                "learning_rate": learning_rate,
                "tokens_per_second": running_tokens / elapsed,
                "peak_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            print(json.dumps(metrics, sort_keys=True), flush=True)
            running_loss = 0.0
            running_tokens = 0
            interval_start = time.perf_counter()

        should_evaluate = (
            completed_step % training_config.evaluation_interval == 0
            or completed_step == step_limit
        )
        if should_evaluate:
            last_validation_loss = evaluate_loss(
                model,
                validation_loader,
                device=device,
                maximum_batches=training_config.evaluation_batches,
                amp=amp_enabled,
            )
            validation_metrics = {
                "step": completed_step,
                "validation_loss": last_validation_loss,
                "validation_perplexity": math.exp(min(20.0, last_validation_loss)),
            }
            print(json.dumps(validation_metrics, sort_keys=True), flush=True)
            if last_validation_loss < best_validation_loss:
                best_validation_loss = last_validation_loss
                save_checkpoint(
                    checkpoint_directory / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=completed_step,
                    best_validation_loss=best_validation_loss,
                    model_config=model.config,
                    training_config=training_config,
                )

        if (
            completed_step % training_config.checkpoint_interval == 0
            or completed_step == step_limit
        ):
            save_checkpoint(
                checkpoint_directory / "latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=completed_step,
                best_validation_loss=best_validation_loss,
                model_config=model.config,
                training_config=training_config,
            )

    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    elapsed_training = max(time.perf_counter() - training_start, 1e-9)
    return {
        "steps": step_limit,
        "final_training_loss": last_loss,
        "final_training_perplexity": math.exp(min(20.0, last_loss)),
        "final_validation_loss": last_validation_loss,
        "final_validation_perplexity": math.exp(min(20.0, last_validation_loss)),
        "best_validation_loss": best_validation_loss,
        "final_learning_rate": final_learning_rate,
        "average_tokens_per_second": total_tokens / elapsed_training,
        "parameter_count": model.parameter_count(),
        "peak_gpu_memory_mib": round(peak_memory, 3),
        "device": str(device),
    }
