"""Standalone visual summaries for prepared data and model experiments."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ai_piano_midi_generator.tokenizer import ExpressiveMidiTokenizer

JsonObject = dict[str, Any]

# Light chart chrome and a CVD-validated categorical order (validate_palette.js:
# first three slots pass all-pairs). Color follows the run name, never its rank.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_REFERENCE = _MUTED  # held-out data is a neutral reference, not a series
_CRITICAL = "#d03b3b"  # status: degenerate sample (always paired with an x marker)

_STYLE: dict[Any, Any] = {  # matplotlib types keys as RcKeyType
    "figure.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": _INK,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 9.5,
    "axes.labelcolor": _INK_SECONDARY,
    "axes.edgecolor": _AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": _AXIS,
    "ytick.color": _AXIS,
    "xtick.labelcolor": _INK_SECONDARY,
    "ytick.labelcolor": _INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
}
_PITCH_CLASSES = ("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B")
_MAXIMUM_CHORD_SIZE = 8


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


def _optional_number(mapping: Mapping[str, object], key: str) -> float | None:
    return _number(mapping, key) if mapping.get(key) is not None else None


def _compact(value: float, _position: object = None) -> str:
    for threshold, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:g}{suffix}"
    return f"{value:g}"


def _styled(builder: Callable[[Path], Figure]) -> Callable[[Path], Figure]:
    """Build a figure inside the shared style so every chart reads as one system."""

    @wraps(builder)
    def build(project_root: Path) -> Figure:
        with plt.rc_context(_STYLE):
            return builder(project_root)

    return build


def _run_colors(project_root: Path) -> dict[str, str]:
    """Assign each report directory a fixed slot by name, shared by all figures."""
    names = sorted(
        {
            path.parent.name
            for pattern in (
                "reports/*/training.json",
                "reports/*/evaluation.json",
                "reports/*/natural_termination.json",
            )
            for path in project_root.glob(pattern)
        }
    )
    if len(names) > len(_SERIES):
        raise ValueError("More model runs than categorical colors; facet instead")
    return dict(zip(names, _SERIES, strict=False))


def _bars(
    axis: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str] | str,
    *,
    horizontal: bool = False,
    value_labels: Sequence[str] | None = None,
) -> None:
    plot = axis.barh if horizontal else axis.bar
    bars = plot(labels, values, 0.6, color=colors)  # 0.6 = bar thickness
    axis.bar_label(
        bars,
        labels=value_labels or [f"{value:,.0f}" for value in values],
        padding=3,
        fontsize=8.5,
        color=_INK_SECONDARY,
    )


@_styled
def dataset_overview_figure(project_root: Path) -> Figure:
    """Show raw inspection outcomes, final file outcomes, and note retention."""
    inventory = read_json_object(project_root / "reports/dataset/inventory.json")
    validation = read_json_object(project_root / "reports/dataset/validation.json")
    raw_classes = _object(
        inventory.get("classification_counts"), "classification_counts"
    )
    final_statuses = _object(validation.get("run_status_counts"), "run_status_counts")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")

    _bars(
        axes[0],
        ["Accepted", "Needs review", "Rejected"],
        [
            _number(raw_classes, "accepted"),
            _number(raw_classes, "needs_review"),
            _number(raw_classes, "rejected"),
        ],
        _SERIES[0],
    )
    axes[0].set_title("Raw inspection")
    axes[0].set_ylabel("MIDI files")

    _bars(
        axes[1],
        ["Usable", "Rejected", "Failed", "Needs review"],
        [
            _number(validation, "final_usable_file_count"),
            _number(final_statuses, "rejected"),
            _number(final_statuses, "failed"),
            _number(final_statuses, "needs_review"),
        ],
        [_SERIES[0], _MUTED, _MUTED, _MUTED],
    )
    axes[1].set_title("After track selection and cleaning")
    axes[1].set_ylabel("MIDI files")

    raw_notes = _number(validation, "raw_note_count")
    processed_notes = _number(validation, "processed_note_count")
    retention = 100.0 * processed_notes / raw_notes if raw_notes else 0.0
    _bars(
        axes[2],
        ["All raw tracks", "Selected piano"],
        [raw_notes, processed_notes],
        [_MUTED, _SERIES[0]],
    )
    axes[2].set_title(f"Notes kept: {retention:.1f}% of all raw notes")
    axes[2].set_ylabel("Notes")

    for axis in axes:
        axis.yaxis.set_major_formatter(FuncFormatter(_compact))
        axis.margins(y=0.12)
    figure.suptitle("Dataset preparation", x=0.01, ha="left")
    return figure


@_styled
def tokenization_overview_figure(project_root: Path) -> Figure:
    """Show linear token-family shares and full-performance sequence lengths."""
    analysis = read_json_object(project_root / "reports/tokenization/analysis.json")
    preparation = read_json_object(
        project_root / "reports/tokenization/dataset_preparation.json"
    )
    rows = read_jsonl_objects(project_root / "data/manifests/tokenized_dataset.jsonl")
    family_counts = _object(
        analysis.get("token_family_distribution"), "token_family_distribution"
    )
    settings = _object(preparation.get("settings"), "settings")
    context_lengths = settings.get("context_lengths")
    if not isinstance(context_lengths, list) or not context_lengths:
        raise ValueError("dataset_preparation settings lack context_lengths")
    context_length = int(context_lengths[0])

    families = {
        "Time shift": _number(family_counts, "time_shift"),
        "Note on": _number(family_counts, "note_on"),
        "Note off": _number(family_counts, "note_off"),
        "Velocity": _number(family_counts, "velocity"),
        "Pedal down": _number(family_counts, "pedal_down"),
        "Pedal up": _number(family_counts, "pedal_up"),
        "BOS / EOS": _number(family_counts, "bos") + _number(family_counts, "eos"),
    }
    total = sum(families.values())
    token_counts = [_number(row, "token_count") for row in rows]
    if not token_counts or not total:
        raise ValueError("Tokenization reports are empty")

    figure, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.4), layout="constrained", width_ratios=(1, 1.15)
    )

    labels = list(reversed(families))
    shares = [100.0 * families[label] / total for label in labels]
    _bars(
        axes[0],
        labels,
        shares,
        _SERIES[0],
        horizontal=True,
        value_labels=[
            f"{share:.1f}%  ({families[label]:,.0f})"
            if share >= 0.1
            else f"<0.1%  ({families[label]:,.0f})"
            for share, label in zip(shares, labels, strict=True)
        ],
    )
    axes[0].set_title("Token families (share of all tokens)")
    axes[0].xaxis.set_major_formatter(PercentFormatter(decimals=0))
    axes[0].set_xlim(0, max(shares) * 1.45)
    axes[0].grid(axis="y", visible=False)
    axes[0].grid(axis="x", visible=True)

    bin_count = min(30, max(10, round(len(token_counts) ** 0.5)))
    axes[1].hist(
        token_counts, bins=bin_count, color=_SERIES[0], edgecolor=_SURFACE, linewidth=1
    )
    axes[1].axvline(context_length, color=_INK_SECONDARY, linewidth=1.2)
    axes[1].annotate(
        f"{context_length:,}-token context",
        (context_length, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        rotation=90,
        ha="left",
        va="top",
        fontsize=8.5,
        color=_INK_SECONDARY,
    )
    longer = sum(count > context_length + 1 for count in token_counts)
    axes[1].text(
        0.98,
        0.96,
        f"{longer} of {len(token_counts)} performances\nspan several context windows",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=_INK_SECONDARY,
    )
    axes[1].set_title("Tokens per complete performance")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Performances")
    axes[1].xaxis.set_major_formatter(FuncFormatter(_compact))

    figure.suptitle(
        f"Expressive event tokenization: {total:,.0f} tokens", x=0.01, ha="left"
    )
    return figure


@_styled
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
        ("file_percent", "Files"),
        ("token_percent", "Tokens"),
        ("tune_group_percent", "Tune groups"),
    ]

    figure, axis = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    positions = np.arange(len(splits), dtype=float)
    width = 0.24
    for index, ((field, label), color) in enumerate(
        zip(metrics, _SERIES, strict=False)
    ):
        values = [
            _number(_object(split_balance.get(split), split), field) for split in splits
        ]
        bars = axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor=_SURFACE,
            linewidth=1,
        )
        texts = axis.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in values],
            padding=3,
            fontsize=8.5,
            color=_INK_SECONDARY,
            zorder=5,
        )
        for text in texts:
            text.set_bbox({"facecolor": _SURFACE, "edgecolor": "none", "pad": 0.6})

    for position, split in zip(positions, splits, strict=True):
        target = _number(target_ratios, split)
        target = target * 100.0 if target <= 1.0 else target
        axis.hlines(
            target,
            position - 1.6 * width,
            position + 1.6 * width,
            color=_INK,
            linewidth=1.4,
            zorder=4,
        )
    axis.set_xticks(positions, ["Train", "Validation", "Test"])
    axis.set_ylabel("Share of dataset")
    axis.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    axis.set_ylim(0, 92)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        [*handles, Line2D([], [], color=_INK, linewidth=1.4)],
        [*labels, "Target share"],
        loc="upper right",
    )
    leakage_free = preparation.get("group_leakage_free")
    suffix = ""
    if leakage_free is True:
        suffix = ": no tune group spans two splits"
    elif leakage_free is False:
        suffix = ": tune-group leakage detected"
    axis.set_title(f"Grouped split balance{suffix}")
    return figure


def _history_points(path: Path, key: str) -> list[tuple[float, float]]:
    """Read (step, value) pairs written by ``train_model``'s history log."""
    if not path.is_file():
        return []
    return [
        (_number(row, "step"), _number(row, key))
        for row in read_jsonl_objects(path)
        if "step" in row and row.get(key) is not None
    ]


@_styled
def training_loss_figure(project_root: Path) -> Figure:
    """Plot learning curves and the comparable held-out test loss per run."""
    training_paths = sorted(project_root.glob("reports/*/training.json"))
    evaluation_paths = sorted(project_root.glob("reports/*/evaluation.json"))
    if not training_paths and not evaluation_paths:
        raise FileNotFoundError(project_root / "reports/*/training.json")
    colors = _run_colors(project_root)

    figure, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.6), layout="constrained", width_ratios=(1.7, 1)
    )

    has_curves = False
    for path in training_paths:
        name = path.parent.name
        report = read_json_object(path)
        color = colors[name]
        history_path = path.with_name("training_history.jsonl")
        training = _history_points(history_path, "training_loss")
        validation = _history_points(history_path, "validation_loss")
        initial = _optional_number(report, "initial_loss")
        if initial is not None:
            # The untrained model is measured on the validation split.
            validation = [(0.0, initial), *validation]
        if training:
            steps, losses = zip(*training, strict=True)
            axes[0].plot(steps, losses, color=color, linewidth=1.0, alpha=0.35)
        if len(validation) > 2:
            steps, losses = zip(*validation, strict=True)
            axes[0].plot(steps, losses, color=color, marker="o", markersize=3.5)
            label = name
        else:
            # Without a per-step history only endpoints exist; never join them
            # with a line that would imply a learning curve.
            endpoints = [*validation]
            endpoints.append(
                (_number(report, "steps"), _number(report, "final_validation_loss"))
            )
            steps, losses = zip(*endpoints, strict=True)
            axes[0].scatter(steps, losses, color=color, s=40, zorder=3)
            label = f"{name} (endpoints only)"
        axes[0].plot([], [], color=color, marker="o", markersize=5, label=label)
        has_curves = has_curves or bool(training) or len(validation) > 2
    if has_curves:
        axes[0].plot([], [], color=_MUTED, linewidth=1.0, alpha=0.6, label="training")
        axes[0].plot(
            [], [], color=_MUTED, marker="o", markersize=3.5, label="validation"
        )
    axes[0].set_title("Learning curves (validation loss as markers)")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Cross-entropy (nats/token)")
    axes[0].xaxis.set_major_formatter(FuncFormatter(_compact))
    axes[0].grid(axis="x", visible=True)
    axes[0].legend(loc="upper right")

    names = [path.parent.name for path in evaluation_paths]
    losses = [
        _number(
            _object(read_json_object(path).get("test_metrics"), "test_metrics"),
            "test_loss",
        )
        for path in evaluation_paths
    ]
    positions = np.arange(len(names), dtype=float)
    axes[1].scatter(
        positions, losses, color=[colors[name] for name in names], s=64, zorder=3
    )
    for position, loss in zip(positions, losses, strict=True):
        axes[1].annotate(
            f"{loss:.3f}",
            (position, loss),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_INK_SECONDARY,
        )
    axes[1].set_xticks(positions, names)
    axes[1].set_xlim(-0.6, len(names) - 0.4)
    if losses:
        spread = max(max(losses) - min(losses), 0.02)
        axes[1].set_ylim(min(losses) - spread, max(losses) + spread)
    axes[1].set_title("Held-out test loss")
    axes[1].set_ylabel("Cross-entropy (nats/token)")

    figure.suptitle("Model training and held-out loss", x=0.01, ha="left")
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


def _natural_termination_samples(project_root: Path) -> dict[str, list[JsonObject]]:
    samples: dict[str, list[JsonObject]] = {}
    for path in sorted(project_root.glob("reports/*/natural_termination.json")):
        values = read_json_object(path).get("samples")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Expected a non-empty samples list in {path}")
        samples[path.parent.name] = [
            _object(sample, "natural_termination sample") for sample in values
        ]
    if not samples:
        raise FileNotFoundError(project_root / "reports/*/natural_termination.json")
    return samples


@_styled
def generated_vs_test_figure(project_root: Path) -> Figure:
    """Compare every run's natural generations with held-out 2,048-token windows."""
    runs = _natural_termination_samples(project_root)
    diagnostics = read_json_object(
        project_root / "reports/baseline_2048/time_shift_diagnostics.json"
    )
    colors = _run_colors(project_root)
    modes = ("unconditional", "prompted")
    metric_specs: Sequence[tuple[str, str, str, float, bool]] = (
        ("generated_elapsed_seconds", "elapsed_token_duration_seconds",
         "Duration (s)", 1.0, True),
        ("generated_notes_per_second", "notes_per_second",
         "Note onsets per second", 1.0, False),
        ("time_shift_token_percentage", "time_shift_token_ratio",
         "Time-shift tokens (%)", 100.0, False),
        ("maximum_silence_seconds", "maximum_silence_seconds",
         "Longest silence (s)", 1.0, True),
    )  # fmt: skip

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    run_names = list(runs)
    offsets = np.linspace(-0.2, 0.2, len(run_names)) if len(run_names) > 1 else [0.0]
    jitter = np.random.default_rng(0)

    for axis, (field, test_field, title, scale, log_scale) in zip(
        axes.flat, metric_specs, strict=True
    ):
        low, median, high = (
            value * scale for value in _test_distribution(diagnostics, test_field)
        )
        axis.axhspan(low, high, color=_REFERENCE, alpha=0.12, linewidth=0, zorder=0)
        axis.axhline(median, color=_REFERENCE, linewidth=1.2, zorder=1)
        for mode_index, mode in enumerate(modes):
            for run_name, offset in zip(run_names, offsets, strict=True):
                samples = [
                    s for s in runs[run_name] if s.get("generation_type") == mode
                ]
                if not samples:
                    continue
                values = np.asarray([_number(s, field) for s in samples])
                degenerate = np.asarray(
                    [
                        bool(
                            _object(
                                s.get("degeneracy_classification"),
                                "degeneracy_classification",
                            ).get("is_degenerate")
                        )
                        for s in samples
                    ],
                    dtype=np.bool_,
                )
                center = mode_index + offset
                axis.boxplot(
                    values,
                    positions=[center],
                    widths=0.16,
                    showfliers=False,
                    medianprops={"color": _INK, "linewidth": 1.6},
                    boxprops={"color": _INK_SECONDARY, "linewidth": 0.9},
                    whiskerprops={"color": _INK_SECONDARY, "linewidth": 0.9},
                    capprops={"color": _INK_SECONDARY, "linewidth": 0.9},
                )
                x_values = center + jitter.uniform(-0.06, 0.06, len(values))
                axis.scatter(
                    x_values[~degenerate],
                    values[~degenerate],
                    s=22,
                    color=colors[run_name],
                    alpha=0.8,
                    edgecolors=_SURFACE,
                    linewidths=0.8,
                    zorder=3,
                )
                axis.scatter(
                    x_values[degenerate],
                    values[degenerate],
                    s=34,
                    marker="x",
                    color=_CRITICAL,
                    linewidths=1.5,
                    zorder=4,
                )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xticks(range(len(modes)), [mode.capitalize() for mode in modes])
        axis.set_xlim(-0.6, len(modes) - 0.4)
        axis.set_title(title)

    sample_counts = {name: len(samples) for name, samples in runs.items()}
    figure.legend(
        handles=[
            *(
                Line2D([], [], marker="o", linestyle="none", color=colors[name],
                       label=f"{name} (n={count})")
                for name, count in sample_counts.items()
            ),
            Line2D([], [], marker="x", linestyle="none", color=_CRITICAL,
                   label="Degenerate (outside train-derived limits)"),
            Patch(facecolor=_REFERENCE, alpha=0.25,
                  label="Held-out test windows: 1st-99th percentile"),
            Line2D([], [], color=_REFERENCE, linewidth=1.2, label="Test median"),
        ],
        loc="outside lower center",
        ncols=3,
    )  # fmt: skip
    figure.suptitle(
        "Natural generations (EOS allowed, ≤2,048 tokens) vs held-out data",
        x=0.01,
        ha="left",
    )
    return figure


def _note_statistics(tokens: Sequence[str]) -> tuple[Counter[int], Counter[int]]:
    """Return note-on pitch counts and notes-per-onset (chord size) counts."""
    pitches: Counter[int] = Counter()
    chord_sizes: Counter[int] = Counter()
    onset_notes = 0
    for token in tokens:
        if token.startswith("NOTE_ON_"):
            pitches[int(token.removeprefix("NOTE_ON_"))] += 1
            onset_notes += 1
        elif token.startswith("TIME_SHIFT_") and onset_notes:
            chord_sizes[min(onset_notes, _MAXIMUM_CHORD_SIZE)] += 1
            onset_notes = 0
    if onset_notes:
        chord_sizes[min(onset_notes, _MAXIMUM_CHORD_SIZE)] += 1
    return pitches, chord_sizes


def _distribution(counts: Counter[int], keys: Sequence[int]) -> np.ndarray:
    values = np.asarray([counts[key] for key in keys], dtype=np.float64)
    total = values.sum()
    return values / total if total else values


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    middle = (first + second) / 2

    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / middle[mask])))

    return (divergence(first) + divergence(second)) / 2


@_styled
def musical_statistics_figure(project_root: Path) -> Figure:
    """Compare generated pitch content, register, and chord sizes with test data."""
    tokenizer = ExpressiveMidiTokenizer()

    def tokens_at(path_value: object) -> tuple[str, ...]:
        if not isinstance(path_value, str):
            raise ValueError("Expected a token_array_path string")
        return tokenizer.ids_to_tokens(
            np.load(project_root / path_value, allow_pickle=False)
        )

    reference_pitches: Counter[int] = Counter()
    reference_chords: Counter[int] = Counter()
    for row in read_jsonl_objects(
        project_root / "data/manifests/tokenized_dataset.jsonl"
    ):
        if row.get("split") == "test":
            pitches, chords = _note_statistics(tokens_at(row.get("token_array_path")))
            reference_pitches.update(pitches)
            reference_chords.update(chords)
    if not reference_pitches:
        raise ValueError("No held-out test notes found")

    generated: dict[str, tuple[Counter[int], Counter[int]]] = {}
    for name, samples in _natural_termination_samples(project_root).items():
        pitches = Counter[int]()
        chords = Counter[int]()
        for sample in samples:
            start = int(_number(sample, "prompt_token_count"))
            stop = start + int(_number(sample, "model_generated_token_count"))
            sample_pitches, sample_chords = _note_statistics(
                tokens_at(sample.get("token_array_path"))[start:stop]
            )
            pitches.update(sample_pitches)
            chords.update(sample_chords)
        generated[name] = (pitches, chords)
    colors = _run_colors(project_root)

    def pitch_classes(counts: Counter[int]) -> Counter[int]:
        folded: Counter[int] = Counter()
        for pitch, count in counts.items():
            folded[pitch % 12] += count
        return folded

    figure, axes = plt.subplots(
        1, 3, figsize=(14, 4.4), layout="constrained", width_ratios=(1.1, 1.3, 0.9)
    )
    reference_label = "Held-out test performances"

    classes = list(range(12))
    reference = _distribution(pitch_classes(reference_pitches), classes)
    axes[0].bar(classes, reference, width=0.6, color=_REFERENCE, alpha=0.45,
                label=reference_label)  # fmt: skip
    for name, (pitches, _) in generated.items():
        values = _distribution(pitch_classes(pitches), classes)
        distance = _jensen_shannon(values, reference)
        axes[0].plot(classes, values, color=colors[name], marker="o", markersize=4.5,
                     label=f"{name} (JS divergence {distance:.3f} bits)")  # fmt: skip
    axes[0].set_xticks(classes, _PITCH_CLASSES)
    axes[0].set_title("Pitch-class distribution")
    axes[0].set_ylabel("Share of note onsets")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # A one-octave moving average removes the black/white-key zigzag so the
    # register envelope is readable; the y value is the mean share per key.
    pitch_range = list(range(21, 109))
    window = np.ones(12) / 12

    def register(counts: Counter[int]) -> np.ndarray:
        return np.convolve(_distribution(counts, pitch_range), window, mode="same")

    axes[1].fill_between(pitch_range, register(reference_pitches), color=_REFERENCE,
                         alpha=0.3, linewidth=0)  # fmt: skip
    for name, (pitches, _) in generated.items():
        axes[1].plot(pitch_range, register(pitches), color=colors[name])
    octave_cs = list(range(24, 109, 12))
    axes[1].set_xticks(octave_cs, [f"C{pitch // 12 - 1}" for pitch in octave_cs])
    axes[1].set_xlim(21, 108)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Register (one-octave moving average, middle C = C4)")
    axes[1].set_ylabel("Share of note onsets per key")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

    sizes = list(range(1, _MAXIMUM_CHORD_SIZE + 1))
    axes[2].bar(sizes, _distribution(reference_chords, sizes), width=0.6,
                color=_REFERENCE, alpha=0.45)  # fmt: skip
    for name, (_, chords) in generated.items():
        axes[2].plot(sizes, _distribution(chords, sizes), color=colors[name],
                     marker="o", markersize=4.5)  # fmt: skip
    axes[2].set_xticks(sizes, [*map(str, sizes[:-1]), f"{_MAXIMUM_CHORD_SIZE}+"])
    axes[2].set_title("Notes struck together")
    axes[2].set_xlabel("Note onsets sharing one 10 ms step")
    axes[2].set_ylabel("Share of onset events")
    axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
                  ncols=3)  # fmt: skip
    figure.suptitle(
        "Musical statistics: generated tokens (prompts excluded) vs held-out test",
        x=0.01,
        ha="left",
    )
    return figure


_FIGURE_BUILDERS: tuple[tuple[str, Callable[[Path], Figure]], ...] = (
    ("dataset_overview", dataset_overview_figure),
    ("tokenization_overview", tokenization_overview_figure),
    ("split_balance", split_balance_figure),
    ("training_loss", training_loss_figure),
    ("generated_vs_test", generated_vs_test_figure),
    ("musical_statistics", musical_statistics_figure),
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
