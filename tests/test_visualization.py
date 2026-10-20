"""Tests for report-driven project visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ai_piano_midi_generator.tokenizer import ExpressiveMidiTokenizer
from ai_piano_midi_generator.visualization import (
    generate_project_figures,
    training_loss_figure,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_generate_project_figures_from_existing_report_shapes(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "reports/dataset/inventory.json",
        {"classification_counts": {"accepted": 8, "needs_review": 1, "rejected": 1}},
    )
    _write_json(
        tmp_path / "reports/dataset/validation.json",
        {
            "final_usable_file_count": 9,
            "raw_note_count": 1_000,
            "processed_note_count": 700,
            "run_status_counts": {
                "processed": 9,
                "rejected": 1,
                "failed": 0,
                "needs_review": 0,
                "skipped": 0,
            },
        },
    )
    _write_json(
        tmp_path / "reports/tokenization/analysis.json",
        {
            "total_token_count": 4_600,
            "token_family_distribution": {
                "bos": 2,
                "velocity": 900,
                "note_on": 1_000,
                "note_off": 1_000,
                "time_shift": 1_500,
                "pedal_down": 99,
                "pedal_up": 97,
                "eos": 2,
            },
        },
    )
    split_balance = {
        split: {
            "file_percent": percent,
            "token_percent": percent,
            "tune_group_percent": percent,
        }
        for split, percent in (("train", 80.0), ("validation", 10.0), ("test", 10.0))
    }
    _write_json(
        tmp_path / "reports/tokenization/dataset_preparation.json",
        {
            "settings": {
                "context_lengths": [2048],
                "target_split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            },
            "group_leakage_free": True,
            "split_balance": split_balance,
        },
    )
    tokenizer = ExpressiveMidiTokenizer()
    performance = tokenizer.tokens_to_ids(
        [
            "BOS",
            "VELOCITY_8",
            "NOTE_ON_60",
            "NOTE_ON_64",
            "TIME_SHIFT_10",
            "NOTE_OFF_60",
            "NOTE_OFF_64",
            "VELOCITY_9",
            "NOTE_ON_67",
            "TIME_SHIFT_5",
            "NOTE_OFF_67",
            "EOS",
        ]
    )
    token_path = tmp_path / "tokens/performance.npy"
    token_path.parent.mkdir(parents=True)
    np.save(token_path, performance)
    manifest_path = tmp_path / "data/manifests/tokenized_dataset.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    "token_count": count,
                    "split": split,
                    "token_array_path": "tokens/performance.npy",
                }
            )
            + "\n"
            for count, split in ((1800, "train"), (2800, "validation"), (3600, "test"))
        ),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "reports/baseline_2048/training.json",
        {
            "context_length": 2048,
            "steps": 100,
            "initial_loss": 6.0,
            "final_training_loss": 2.2,
            "best_validation_loss": 2.5,
            "final_validation_loss": 2.6,
        },
    )
    _write_json(
        tmp_path / "reports/baseline_2048/evaluation.json",
        {
            "checkpoint_best_validation_loss": 2.4,
            "test_metrics": {"test_loss": 2.6},
        },
    )
    samples = []
    for mode in ("unconditional", "prompted"):
        for offset in (0.0, 1.0):
            samples.append(
                {
                    "generation_type": mode,
                    "generated_elapsed_seconds": 50.0 + offset,
                    "generated_notes_per_second": 7.0 + offset,
                    "time_shift_token_percentage": 36.0 + offset,
                    "maximum_silence_seconds": 2.0 + offset,
                    "degeneracy_classification": {"is_degenerate": offset > 0.0},
                    "token_array_path": "tokens/performance.npy",
                    "prompt_token_count": 1,
                    "model_generated_token_count": 10,
                }
            )
    _write_json(
        tmp_path / "reports/baseline_2048/natural_termination.json",
        {"samples": samples},
    )
    distributions = {
        "elapsed_token_duration_seconds": {"p01": 35.0, "median": 50.0, "p99": 90.0},
        "notes_per_second": {"p01": 4.0, "median": 7.0, "p99": 12.0},
        "time_shift_token_ratio": {"p01": 0.33, "median": 0.36, "p99": 0.39},
        "maximum_silence_seconds": {"p01": 0.5, "median": 1.5, "p99": 7.0},
    }
    _write_json(
        tmp_path / "reports/baseline_2048/time_shift_diagnostics.json",
        {
            "dataset_statistics": {
                "test": {
                    "nonoverlapping_2048_token_windows": {
                        "metric_distributions": distributions
                    }
                }
            }
        },
    )

    output_directory = tmp_path / "reports/figures"
    result = generate_project_figures(
        tmp_path, output_directory, formats=("png",), dpi=72, strict=True
    )

    assert result.skipped_plots == ()
    assert {path.name for path in result.generated_files} == {
        "dataset_overview.png",
        "tokenization_overview.png",
        "split_balance.png",
        "training_loss.png",
        "generated_vs_test.png",
        "musical_statistics.png",
    }
    assert all(path.stat().st_size > 1_000 for path in result.generated_files)
    manifest = json.loads((output_directory / "manifest.json").read_text())
    assert manifest["skipped_plots"] == []
    assert len(manifest["generated_files"]) == 6


def test_training_loss_figure_draws_curves_only_from_logged_history(
    tmp_path: Path,
) -> None:
    report = {
        "steps": 20,
        "initial_loss": 6.0,
        "final_training_loss": 2.2,
        "best_validation_loss": 2.5,
        "final_validation_loss": 2.6,
    }
    _write_json(tmp_path / "reports/baseline_2048/training.json", report)
    (tmp_path / "reports/baseline_2048/training_history.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for step in (10, 20)
            for row in (
                {"step": step, "training_loss": 3.0 - step / 20},
                {"step": step, "validation_loss": 3.1 - step / 20},
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "reports/eos_2048/training.json",
        {**report, "initial_loss": None, "final_validation_loss": 2.5},
    )

    figure = training_loss_figure(tmp_path)

    legend = {text.get_text() for text in figure.axes[0].get_legend().get_texts()}
    assert legend == {
        "baseline_2048",
        "eos_2048 (endpoints only)",
        "training",
        "validation",
    }
    validation_curve = [
        line for line in figure.axes[0].get_lines() if len(line.get_xdata()) == 3
    ]
    assert [list(line.get_xdata()) for line in validation_curve] == [[0.0, 10, 20]]
