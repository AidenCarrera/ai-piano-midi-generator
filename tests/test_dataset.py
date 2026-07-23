from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_piano_midi_generator.dataset import TokenChunkDataset, collate_token_chunks
from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    PEDAL_UP_TOKEN,
    ExpressiveMidiTokenizer,
)


def write_array(
    path: Path, tokenizer: ExpressiveMidiTokenizer, tokens: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokenizer.tokens_to_ids(tokens), allow_pickle=False)


def test_dataset_loads_only_requested_split_and_reconstructs_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = ExpressiveMidiTokenizer()
    train_tokens = (
        BOS_TOKEN,
        PEDAL_DOWN_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_60",
        "TIME_SHIFT_10",
        "TIME_SHIFT_10",
        "TIME_SHIFT_10",
        "TIME_SHIFT_10",
        "NOTE_OFF_60",
        PEDAL_UP_TOKEN,
        EOS_TOKEN,
    )
    validation_tokens = (
        BOS_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_64",
        "TIME_SHIFT_10",
        "NOTE_OFF_64",
        EOS_TOKEN,
    )
    train_path = tmp_path / "train.npy"
    validation_path = tmp_path / "validation.npy"
    write_array(train_path, tokenizer, train_tokens)
    write_array(validation_path, tokenizer, validation_tokens)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_file": "train.mid",
                        "tune_group": "train-tune",
                        "split": "train",
                        "token_array_path": str(train_path),
                    }
                ),
                json.dumps(
                    {
                        "source_file": "validation.mid",
                        "tune_group": "validation-tune",
                        "split": "validation",
                        "token_array_path": str(validation_path),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = TokenChunkDataset(manifest, "train", tokenizer, context_length=8)

    assert dataset.performance_files == ("train.mid",)
    assert len(dataset) == 2
    unsafe = dataset.chunks[1]
    reconstructed = tuple(
        tokenizer.vocabulary[token_id] for token_id in unsafe.reconstruction_ids
    )
    assert unsafe.plan.state_reconstruction_required is True
    assert reconstructed == (
        BOS_TOKEN,
        PEDAL_DOWN_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_60",
    )
    assert dataset.cached_token_bytes > 0

    def reject_reopen(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError(f"np.load called after dataset setup: {args}, {kwargs}")

    monkeypatch.setattr(np, "load", reject_reopen)
    inputs, targets = dataset[1]
    assert inputs.numel() <= 8
    assert targets.shape == inputs.shape


def test_chunk_collation_masks_right_padding() -> None:
    first = (np.array([1, 2, 3]), np.array([2, 3, 4]))
    second = (np.array([1, 2]), np.array([2, 3]))

    batch = [
        (torch.from_numpy(inputs), torch.from_numpy(targets))
        for inputs, targets in (first, second)
    ]
    inputs, targets = collate_token_chunks(batch, pad_token_id=0)

    assert inputs.tolist() == [[1, 2, 3], [1, 2, 0]]
    assert targets.tolist() == [[2, 3, 4], [2, 3, -100]]


def test_chunk_collation_supports_fixed_compiler_friendly_length() -> None:
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        (torch.tensor([5, 6]), torch.tensor([6, 7])),
    ]

    inputs, targets = collate_token_chunks(
        batch,
        pad_token_id=0,
        padded_length=8,
    )

    assert inputs.shape == (2, 8)
    assert targets.shape == (2, 8)
    assert inputs[0].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    assert targets[1].tolist() == [6, 7, -100, -100, -100, -100, -100, -100]


def test_chunk_collation_rejects_fixed_length_shorter_than_batch() -> None:
    batch = [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]))]

    with pytest.raises(ValueError, match="shorter than batch maximum"):
        collate_token_chunks(batch, padded_length=2)
