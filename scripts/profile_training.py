"""Profile representative eager CUDA training steps with torch.profiler."""

from __future__ import annotations

import argparse
import json
import platform
import time
import tomllib
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function
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
    build_optimizer,
    configure_determinism,
    select_device,
    set_random_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_baseline_2048.toml"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("artifacts/profiler/training_trace.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/benchmarks/training_profile.json"),
    )
    return parser


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Missing TOML table: {name}")
    return value


def _next_batch(
    iterator: Any,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _training_step(
    *,
    model: PianoTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    iterator: Any,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = torch.zeros((), device=device, dtype=torch.float32)
    step_tokens = torch.zeros((), device=device, dtype=torch.int64)
    with record_function("training_step"):
        for _ in range(config.gradient_accumulation_steps):
            with record_function("data_loading"):
                (inputs, targets), iterator = _next_batch(iterator, loader)
            with record_function("host_to_device"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            step_tokens.add_((targets != -100).sum())
            with record_function("forward"):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    _, loss = model(inputs, targets)
                assert loss is not None
                scaled_loss = loss / config.gradient_accumulation_steps
            with record_function("backward"):
                scaled_loss_tensor = scaler.scale(scaled_loss)
                assert isinstance(scaled_loss_tensor, torch.Tensor)
                scaled_loss_tensor.backward()
            accumulated_loss.add_(loss.detach())

        with record_function("gradient_clipping"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )
        with record_function("optimizer_step"):
            scaler.step(optimizer)
            scaler.update()
    return iterator, accumulated_loss, step_tokens


def _event_metric(event: Any, name: str) -> float:
    value = getattr(event, name, 0.0)
    return float(value) if value is not None else 0.0


def _event_rows(profiler: Any, *, limit: int = 30) -> list[dict[str, object]]:
    events = [
        event
        for event in profiler.key_averages(group_by_input_shape=False)
        if str(event.key).startswith("aten::")
    ]
    events.sort(
        key=lambda event: _event_metric(event, "self_device_time_total"),
        reverse=True,
    )
    total_self_device_time = sum(
        _event_metric(event, "self_device_time_total") for event in events
    )
    rows: list[dict[str, object]] = []
    for event in events[:limit]:
        self_device_time = _event_metric(event, "self_device_time_total")
        rows.append(
            {
                "name": str(event.key),
                "calls": int(event.count),
                "self_device_time_us": self_device_time,
                "device_time_total_us": _event_metric(
                    event, "device_time_total"
                ),
                "self_device_percent": (
                    100.0 * self_device_time / total_self_device_time
                    if total_self_device_time
                    else 0.0
                ),
                "self_cpu_time_us": _event_metric(
                    event, "self_cpu_time_total"
                ),
                "device_memory_usage_bytes": int(
                    _event_metric(event, "device_memory_usage")
                ),
            }
        )
    return rows


def _range_rows(profiler: Any) -> list[dict[str, object]]:
    wanted = {
        "training_step",
        "data_loading",
        "host_to_device",
        "forward",
        "backward",
        "gradient_clipping",
        "optimizer_step",
    }
    ranges: dict[str, dict[str, object]] = {}
    for event in profiler.key_averages(group_by_input_shape=False):
        if event.key not in wanted:
            continue
        name = str(event.key)
        row = ranges.setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "cpu_time_total_us": 0.0,
                "device_time_total_us": 0.0,
            },
        )
        calls = row["calls"]
        cpu_time = row["cpu_time_total_us"]
        device_time = row["device_time_total_us"]
        assert isinstance(calls, int)
        assert isinstance(cpu_time, (int, float))
        assert isinstance(device_time, (int, float))
        row["calls"] = max(calls, int(event.count))
        row["cpu_time_total_us"] = max(
            float(cpu_time),
            _event_metric(event, "cpu_time_total"),
        )
        row["device_time_total_us"] = max(
            float(device_time),
            _event_metric(event, "device_time_total"),
        )
    return sorted(ranges.values(), key=lambda row: str(row["name"]))


def _hotspot_summary(profiler: Any, profile_steps: int) -> dict[str, object]:
    events: dict[str, list[Any]] = {}
    for event in profiler.key_averages(group_by_input_shape=False):
        events.setdefault(str(event.key), []).append(event)

    def maximum_metric(key: str, metric: str) -> float:
        return max(
            (_event_metric(event, metric) for event in events.get(key, [])),
            default=0.0,
        )

    def summed_self_device_time(*keys: str) -> float:
        return sum(maximum_metric(key, "self_device_time_total") for key in keys)

    training_step_device_time = maximum_metric(
        "training_step", "device_time_total"
    )

    def component(device_time_us: float) -> dict[str, float]:
        return {
            "device_time_total_us": device_time_us,
            "device_time_per_step_us": device_time_us / profile_steps,
            "training_step_device_percent": (
                100.0 * device_time_us / training_step_device_time
                if training_step_device_time
                else 0.0
            ),
        }

    return {
        "training_step_device_time_per_step_us": (
            training_step_device_time / profile_steps
        ),
        "attention": component(
            maximum_metric(
                "aten::_efficient_attention_forward", "device_time_total"
            )
            + maximum_metric(
                "aten::_efficient_attention_backward", "device_time_total"
            )
        ),
        "matrix_multiplication": component(
            summed_self_device_time("aten::mm", "aten::addmm")
        ),
        "gelu": component(
            summed_self_device_time("aten::gelu", "aten::gelu_backward")
        ),
        "layer_norm": component(
            summed_self_device_time(
                "aten::native_layer_norm",
                "aten::native_layer_norm_backward",
            )
        ),
        "gradient_clipping": component(
            maximum_metric("gradient_clipping", "device_time_total")
        ),
        "optimizer_step": component(
            maximum_metric("optimizer_step", "device_time_total")
        ),
        "host_to_device": component(
            maximum_metric("host_to_device", "device_time_total")
        ),
        "data_loading_cpu_time_per_step_us": (
            maximum_metric("data_loading", "cpu_time_total") / profile_steps
        ),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup_steps < 0 or args.profile_steps <= 0:
        raise ValueError("warmup steps must be non-negative and profile steps positive")

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    data_config = _mapping(raw.get("data"), "data")
    model_config = TransformerConfig(**_mapping(raw.get("model"), "model"))
    training_config = TrainingConfig(**_mapping(raw.get("training"), "training"))
    manifest = Path(str(data_config["manifest"]))
    context_length = int(data_config["context_length"])
    if context_length != model_config.context_length:
        raise ValueError("data.context_length and model.context_length must match")

    device = select_device(args.device)
    set_random_seed(training_config.seed)
    configure_determinism(training_config.deterministic)
    tokenizer = ExpressiveMidiTokenizer()
    dataset = TokenChunkDataset(
        manifest,
        "train",
        tokenizer,
        context_length=context_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.data_loader_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=training_config.data_loader_workers > 0,
        generator=torch.Generator().manual_seed(training_config.seed),
        collate_fn=partial(
            collate_token_chunks,
            pad_token_id=default_pad_token_id(tokenizer),
            padded_length=context_length,
        ),
    )
    model = PianoTransformer(model_config).to(device).train()
    optimizer = build_optimizer(model, training_config)
    amp_enabled = training_config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if training_config.compile_model:
        model.compile(dynamic=False, mode=training_config.compile_mode)

    iterator = iter(loader)
    last_loss = torch.full((), float("nan"), device=device)
    last_tokens = torch.zeros((), dtype=torch.int64, device=device)
    for _ in range(args.warmup_steps):
        iterator, last_loss, last_tokens = _training_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            iterator=iterator,
            loader=loader,
            config=training_config,
            device=device,
            amp_enabled=amp_enabled,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    wall_started = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler:
        for _ in range(args.profile_steps):
            iterator, last_loss, last_tokens = _training_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                iterator=iterator,
                loader=loader,
                config=training_config,
                device=device,
                amp_enabled=amp_enabled,
            )
            profiler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    profiled_wall_seconds = time.perf_counter() - wall_started

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))
    report: dict[str, object] = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "settings": {
            "config": args.config.as_posix(),
            "batch_size": training_config.batch_size,
            "gradient_accumulation_steps": (
                training_config.gradient_accumulation_steps
            ),
            "context_length": context_length,
            "amp": amp_enabled,
            "fused_adamw": bool(optimizer.defaults.get("fused", False)),
            "gradient_clip_norm": training_config.gradient_clip_norm,
            "compile_model": training_config.compile_model,
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
        },
        "profile": {
            "wall_seconds_including_profiler_overhead": profiled_wall_seconds,
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            "last_loss": float(last_loss.item()),
            "last_step_tokens": int(last_tokens.item()),
            "ranges": _range_rows(profiler),
            "hotspot_summary": _hotspot_summary(
                profiler, args.profile_steps
            ),
            "top_self_device_operations": _event_rows(profiler),
        },
        "artifacts": {
            "chrome_trace": args.trace.as_posix(),
        },
        "interpretation_note": (
            "Profiler instrumentation adds overhead; use CUDA-time shares to find "
            "hotspots, not the profiled wall time as a throughput benchmark."
        ),
    }
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
