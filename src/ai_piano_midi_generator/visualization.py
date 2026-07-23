"""Standalone visual summaries for prepared data and model experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

JsonObject = dict[str, Any]

_BLUE = "#4477AA"
_CYAN = "#66CCEE"
_GREEN = "#228833"
_YELLOW = "#CCBB44"
_RED = "#EE6677"
_PURPLE = "#AA3377"
_GRAY = "#777777"


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """Files created and plots skipped because an input was unavailable."""

    generated_files: tuple[Path, ...]
    skipped_plots: tuple[str, ...]


def read_json_object(path: Path) -> JsonObject:
    """Read a JSON object and reject other top-level JSON values."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(JsonObject, value)


def read_jsonl_objects(path: Path) -> list[JsonObject]:
    """Read non-empty JSONL records as objects."""
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict) or not all(
                isinstance(key, str) for key in value
            ):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(cast(JsonObject, value))
    return rows


def _object(value: object, description: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected an object for {description}")
    return cast(JsonObject, value)


def _number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric field {key!r}")
    return float(value)


def _format_count(value: float) -> str:
    return f"{value:,.0f}"


def _style_axis(axis: Axes, *, grid_axis: Literal["both", "x", "y"] = "y") -> None:
    axis.grid(True, axis=grid_axis, alpha=0.22, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


def _label_bars(axis: Axes, bars: Any, *, percent: bool = False) -> None:
    labels = [
        f"{bar.get_height():.1f}%" if percent else _format_count(bar.get_height())
        for bar in bars
    ]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)


def dataset_overview_figure(project_root: Path) -> Figure:
    """Show raw inspection outcomes, final file outcomes, and note retention."""
    inventory = read_json_object(project_root / "reports/dataset/inventory.json")
    validation = read_json_object(project_root / "reports/dataset/validation.json")
    raw_classes = _object(
        inventory.get("classification_counts"), "classification_counts"
    )
    final_statuses = _object(validation.get("run_status_counts"), "run_status_counts")

    figure, axes_value = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    axes = np.asarray(axes_value).ravel().tolist()

    raw_labels = ["Accepted", "Review", "Rejected"]
    raw_values = [
        _number(raw_classes, "accepted"),
        _number(raw_classes, "needs_review"),
        _number(raw_classes, "rejected"),
    ]
    bars = axes[0].bar(raw_labels, raw_values, color=[_GREEN, _YELLOW, _RED])
    _label_bars(axes[0], bars)
    axes[0].set_title("Raw inspection")
    axes[0].set_ylabel("MIDI files")
    _style_axis(axes[0])

    final_labels = ["Usable", "Rejected", "Failed", "Review"]
    final_values = [
        _number(validation, "final_usable_file_count"),
        _number(final_statuses, "rejected"),
        _number(final_statuses, "failed"),
        _number(final_statuses, "needs_review"),
    ]
    bars = axes[1].bar(
        final_labels,
        final_values,
        color=[_GREEN, _RED, _PURPLE, _YELLOW],
    )
    _label_bars(axes[1], bars)
    axes[1].set_title("Final cleaning outcome")
    axes[1].tick_params(axis="x", rotation=20)
    _style_axis(axes[1])

    raw_notes = _number(validation, "raw_note_count")
    processed_notes = _number(validation, "processed_note_count")
    bars = axes[2].bar(
        ["Raw tracks", "Clean piano"],
        [raw_notes, processed_notes],
        color=[_GRAY, _BLUE],
    )
    _label_bars(axes[2], bars)
    retention = 100.0 * processed_notes / raw_notes if raw_notes else 0.0
    axes[2].text(
        0.5,
        0.86,
        f"{retention:.1f}% of raw notes retained",
        ha="center",
        va="top",
        transform=axes[2].transAxes,
        fontsize=9,
    )
    axes[2].set_title("Note volume")
    axes[2].set_ylabel("Notes")
    _style_axis(axes[2])

    figure.suptitle("Dataset preparation overview", fontsize=15)
    return figure


def tokenization_overview_figure(project_root: Path) -> Figure:
    """Show token-family composition and full-performance sequence lengths."""
    analysis = read_json_object(project_root / "reports/tokenization/analysis.json")
    active_training = read_json_object(
        project_root / "reports/baseline_2048/training.json"
    )
    rows = read_jsonl_objects(project_root / "data/manifests/tokenized_dataset.jsonl")
    family_counts = _object(
        analysis.get("token_family_distribution"), "token_family_distribution"
    )

    family_keys = [
        "bos",
        "velocity",
        "note_on",
        "note_off",
        "time_shift",
        "pedal_down",
        "pedal_up",
        "eos",
    ]
    family_labels = [
        "BOS",
        "Velocity",
        "Note on",
        "Note off",
        "Time shift",
        "Pedal down",
        "Pedal up",
        "EOS",
    ]
    family_values = [_number(family_counts, key) for key in family_keys]
    token_counts = [_number(row, "token_count") for row in rows]
    if not token_counts:
        raise ValueError("Tokenized dataset manifest is empty")

    figure, axes_value = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    axes = np.asarray(axes_value).ravel().tolist()

    bars = axes[0].bar(family_labels, family_values, color=_BLUE)
    _label_bars(axes[0], bars)
    axes[0].set_yscale("log")
    axes[0].set_title("Token-family distribution")
    axes[0].set_ylabel("Tokens (log scale)")
    axes[0].tick_params(axis="x", rotation=35)
    _style_axis(axes[0])

    bin_count = min(24, max(10, round(len(token_counts) ** 0.5)))
    axes[1].hist(token_counts, bins=bin_count, color=_CYAN, edgecolor="white")
    active_context = _number(active_training, "context_length")
    axes[1].axvline(
        active_context,
        color=_RED,
        linewidth=1.5,
        linestyle="--",
        label=f"Active {int(active_context):,}-token context",
    )
    axes[1].set_title("Full-performance sequence lengths")
    axes[1].set_xlabel("Tokens per MIDI file")
    axes[1].set_ylabel("Files")
    axes[1].legend(frameon=False, fontsize=8)
    _style_axis(axes[1])

    figure.suptitle(
        f"Expressive tokenization — {_format_count(_number(analysis, 'total_token_count'))} tokens",
        fontsize=15,
    )
    return figure


def split_balance_figure(project_root: Path) -> Figure:
    """Compare file, token, and tune-group proportions with split targets."""
    preparation = read_json_object(
        project_root / "reports/tokenization/dataset_preparation.json"
    )
    split_balance = _object(preparation.get("split_balance"), "split_balance")
    settings = _object(preparation.get("settings"), "settings")
    target_ratios = _object(settings.get("target_split_ratios"), "target_split_ratios")
    splits = ["train", "validation", "test"]
    metrics = [
        ("file_percent", "Files", _BLUE),
        ("token_percent", "Tokens", _GREEN),
        ("tune_group_percent", "Tune groups", _PURPLE),
    ]

    figure, axis = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    positions = np.arange(len(splits), dtype=float)
    width = 0.23
    for metric_index, (field, label, color) in enumerate(metrics):
        values = [
            _number(_object(split_balance.get(split), split), field) for split in splits
        ]
        bars = axis.bar(
            positions + (metric_index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        _label_bars(axis, bars, percent=True)

    targets = []
    for split in splits:
        target = _number(target_ratios, split)
        targets.append(target * 100.0 if target <= 1.0 else target)
    axis.scatter(
        positions,
        targets,
        marker="D",
        s=52,
        color="black",
        label="Target",
        zorder=4,
    )
    axis.set_xticks(positions, ["Train", "Validation", "Test"])
    axis.set_ylabel("Dataset share (%)")
    axis.set_title("Grouped split balance (no tune leakage)")
    axis.legend(frameon=False, loc="upper right")
    _style_axis(axis)
    return figure


def training_loss_figure(project_root: Path) -> Figure:
    """Visualize available training endpoints and held-out loss measurements."""
    training_paths = sorted(project_root.glob("reports/*/training.json"))
    evaluation_paths = sorted(project_root.glob("reports/*/evaluation.json"))
    if not training_paths and not evaluation_paths:
        raise FileNotFoundError(project_root / "reports/*/training.json")

    training_reports = {
        path.parent.name: read_json_object(path) for path in training_paths
    }
    evaluation_reports = {
        path.parent.name: read_json_object(path) for path in evaluation_paths
    }
    run_names = sorted(set(training_reports) | set(evaluation_reports))
    colors = [_BLUE, _GREEN, _PURPLE, _YELLOW]

    figure, axes_value = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    axes = np.asarray(axes_value).ravel().tolist()

    for index, (name, report) in enumerate(training_reports.items()):
        steps = _number(report, "steps")
        final = _number(report, "final_training_loss")
        initial_value = report.get("initial_loss")
        initial = (
            float(initial_value)
            if isinstance(initial_value, (int, float))
            and not isinstance(initial_value, bool)
            else None
        )
        x_values = [steps] if initial is None else [0.0, steps]
        y_values = [final] if initial is None else [initial, final]
        axes[0].plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            color=colors[index % len(colors)],
            label=name,
        )
        if initial is not None:
            axes[0].annotate(
                f"{initial:.2f}",
                (0.0, initial),
                xytext=(4, 5 + index * 12),
                textcoords="offset points",
                fontsize=8,
            )
        axes[0].annotate(
            f"{final:.2f}",
            (steps, final),
            xytext=(4, 5 + index * 12),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_title("Training-loss endpoints")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Cross-entropy loss")
    if training_reports:
        axes[0].legend(frameon=False)
    axes[0].text(
        0.02,
        0.03,
        "Only summary endpoints were retained; this is not a full learning curve.",
        transform=axes[0].transAxes,
        fontsize=8,
        color=_GRAY,
    )
    _style_axis(axes[0])

    x_positions = np.arange(len(run_names), dtype=float)
    measurement_specs = [
        ("Best validation", -0.24, _GREEN),
        ("Final validation", 0.0, _BLUE),
        ("Held-out test", 0.24, _RED),
    ]
    run_measurements: dict[str, dict[str, float]] = {}
    for name in run_names:
        training_report = training_reports.get(name)
        evaluation_report = evaluation_reports.get(name)
        values: dict[str, float] = {}
        if training_report is not None:
            values["Best validation"] = _number(training_report, "best_validation_loss")
            values["Final validation"] = _number(
                training_report, "final_validation_loss"
            )
        if evaluation_report is not None:
            if "Best validation" not in values:
                values["Best validation"] = _number(
                    evaluation_report, "checkpoint_best_validation_loss"
                )
            test_metrics = _object(
                evaluation_report.get("test_metrics"), "test_metrics"
            )
            values["Held-out test"] = _number(test_metrics, "test_loss")
        run_measurements[name] = values
    maximum_loss = 0.0
    legend_handles: list[Patch] = []
    for label, offset, color in measurement_specs:
        available = [
            (run_index, run_measurements[name][label])
            for run_index, name in enumerate(run_names)
            if label in run_measurements[name]
        ]
        if not available:
            continue
        positions = np.asarray([index + offset for index, _ in available])
        losses = [loss for _, loss in available]
        maximum_loss = max(maximum_loss, max(losses))
        bars = axes[1].bar(positions, losses, width=0.2, color=color)
        axes[1].bar_label(
            bars,
            labels=[f"{loss:.4f}" for loss in losses],
            padding=3,
            fontsize=8,
        )
        legend_handles.append(Patch(facecolor=color, label=label))
    axes[1].set_xticks(x_positions, run_names)
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_ylim(0.0, maximum_loss * 1.15 if maximum_loss else 1.0)
    axes[1].set_title("Available validation and test losses")
    axes[1].legend(handles=legend_handles, frameon=False, fontsize=8)
    _style_axis(axes[1])

    figure.suptitle("Active model loss summary", fontsize=15)
    return figure


def _test_distribution(
    diagnostics: JsonObject, metric_name: str
) -> tuple[float, float, float]:
    dataset_statistics = _object(
        diagnostics.get("dataset_statistics"), "dataset_statistics"
    )
    test_statistics = _object(dataset_statistics.get("test"), "test")
    windows = _object(
        test_statistics.get("nonoverlapping_2048_token_windows"),
        "nonoverlapping_2048_token_windows",
    )
    distributions = _object(windows.get("metric_distributions"), "metric_distributions")
    metric = _object(distributions.get(metric_name), metric_name)
    return (
        _number(metric, "p01"),
        _number(metric, "median"),
        _number(metric, "p99"),
    )


def generated_vs_test_figure(project_root: Path) -> Figure:
    """Compare natural generations with length-matched held-out test windows."""
    natural = read_json_object(
        project_root / "reports/baseline_2048/natural_termination.json"
    )
    diagnostics = read_json_object(
        project_root / "reports/baseline_2048/time_shift_diagnostics.json"
    )
    sample_values = natural.get("samples")
    if not isinstance(sample_values, list):
        raise ValueError("Expected natural_termination.samples to be a list")
    samples = [
        _object(sample, "natural_termination sample") for sample in sample_values
    ]
    modes = ["unconditional", "prompted"]
    metric_specs: Sequence[tuple[str, str, str, Callable[[float], float], bool]] = [
        (
            "generated_elapsed_seconds",
            "elapsed_token_duration_seconds",
            "Elapsed duration (s)",
            lambda value: value,
            True,
        ),
        (
            "generated_notes_per_second",
            "notes_per_second",
            "Notes per second",
            lambda value: value,
            False,
        ),
        (
            "time_shift_token_percentage",
            "time_shift_token_ratio",
            "Time-shift tokens (%)",
            lambda value: value * 100.0,
            False,
        ),
        (
            "maximum_silence_seconds",
            "maximum_silence_seconds",
            "Maximum silence (s)",
            lambda value: value,
            True,
        ),
    ]

    figure, axes_value = plt.subplots(2, 2, figsize=(11.5, 8.2), layout="constrained")
    axes = np.asarray(axes_value).ravel().tolist()
    positions = [1.0, 2.0]
    mode_colors = [_BLUE, _PURPLE]

    for axis, (sample_field, test_field, label, test_transform, log_scale) in zip(
        axes, metric_specs, strict=True
    ):
        grouped_values: list[list[float]] = []
        grouped_samples: list[list[JsonObject]] = []
        for mode in modes:
            mode_samples = [
                sample for sample in samples if sample.get("generation_type") == mode
            ]
            grouped_samples.append(mode_samples)
            grouped_values.append(
                [_number(sample, sample_field) for sample in mode_samples]
            )
        if any(not values for values in grouped_values):
            raise ValueError(f"Missing generated values for {sample_field}")

        boxes = axis.boxplot(
            grouped_values,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            tick_labels=["Unconditional", "Prompted"],
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box, color in zip(boxes["boxes"], mode_colors, strict=True):
            box.set_facecolor(color)
            box.set_alpha(0.65)

        test_low, test_median, test_high = _test_distribution(diagnostics, test_field)
        test_low = test_transform(test_low)
        test_median = test_transform(test_median)
        test_high = test_transform(test_high)
        axis.axhspan(test_low, test_high, color=_GREEN, alpha=0.14, zorder=0)
        axis.axhline(test_median, color=_GREEN, linewidth=1.8, linestyle="--")

        for position, mode_samples in zip(positions, grouped_samples, strict=True):
            degenerate_values = [
                _number(sample, sample_field)
                for sample in mode_samples
                if bool(
                    _object(
                        sample.get("degeneracy_classification"),
                        "degeneracy_classification",
                    ).get("is_degenerate")
                )
            ]
            if degenerate_values:
                jitter = np.linspace(-0.08, 0.08, len(degenerate_values))
                axis.scatter(
                    position + jitter,
                    degenerate_values,
                    marker="x",
                    s=42,
                    linewidth=1.5,
                    color=_RED,
                    zorder=4,
                )

        if (
            log_scale
            and min(test_low, *(value for group in grouped_values for value in group))
            > 0
        ):
            axis.set_yscale("log")
        axis.set_title(label)
        _style_axis(axis)

    axes[0].legend(
        handles=[
            Patch(facecolor=_GREEN, alpha=0.14, label="Test p01-p99"),
            Line2D([], [], color=_GREEN, linestyle="--", label="Test median"),
            Line2D(
                [],
                [],
                color=_RED,
                marker="x",
                linestyle="none",
                label="Degenerate sample",
            ),
        ],
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Natural generations versus held-out 2,048-token windows", fontsize=15
    )
    return figure


_FIGURE_BUILDERS: tuple[tuple[str, Callable[[Path], Figure]], ...] = (
    ("dataset_overview", dataset_overview_figure),
    ("tokenization_overview", tokenization_overview_figure),
    ("split_balance", split_balance_figure),
    ("training_loss", training_loss_figure),
    ("generated_vs_test", generated_vs_test_figure),
)


def generate_project_figures(
    project_root: Path,
    output_directory: Path,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 160,
    strict: bool = False,
) -> VisualizationResult:
    """Build and save all available project figures without mutating source data."""
    normalized_formats = tuple(dict.fromkeys(format_.lower() for format_ in formats))
    unsupported = set(normalized_formats) - {"png", "svg"}
    if unsupported:
        raise ValueError(f"Unsupported figure formats: {sorted(unsupported)}")
    if not normalized_formats:
        raise ValueError("At least one output format is required")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    project_root = project_root.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    skipped: list[str] = []
    for name, builder in _FIGURE_BUILDERS:
        try:
            figure = builder(project_root)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if strict:
                raise
            skipped.append(f"{name}: {exc}")
            continue
        try:
            for format_ in normalized_formats:
                output_path = output_directory / f"{name}.{format_}"
                figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
                generated.append(output_path)
        finally:
            plt.close(figure)

    result = VisualizationResult(tuple(generated), tuple(skipped))
    manifest = {
        "generated_files": [path.as_posix() for path in result.generated_files],
        "skipped_plots": list(result.skipped_plots),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
