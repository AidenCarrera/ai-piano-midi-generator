"""Split-aware token chunk loading with explicit musical state reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ai_piano_midi_generator.data.preparation import ChunkPlan, plan_state_safe_chunks
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class ChunkReference:
    token_array_path: Path
    source_file: str
    tune_group: str
    split: Split
    plan: ChunkPlan
    reconstruction_ids: tuple[int, ...]


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Tokenized dataset manifest not found: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path} line {line_number}")
        rows.append(value)
    return rows


def _string_field(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Manifest row is missing string {key}")
    return value


class TokenChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load chunks from exactly one grouped dataset split."""

    def __init__(
        self,
        manifest_path: Path,
        split: Split,
        tokenizer: ExpressiveMidiTokenizer,
        *,
        context_length: int = 1024,
        maximum_performances: int | None = None,
        maximum_chunks: int | None = None,
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown dataset split: {split}")
        if context_length < 8:
            raise ValueError("context_length must be at least 8")
        self.split = split
        self.context_length = context_length
        self.tokenizer = tokenizer
        rows = [
            row
            for row in _read_manifest(manifest_path)
            if _string_field(row, "split") == split
        ]
        if maximum_performances is not None:
            rows = rows[:maximum_performances]
        if not rows:
            raise ValueError(f"No performances found for split {split}")

        chunks: list[ChunkReference] = []
        for row in rows:
            token_path_value = _string_field(row, "token_array_path")
            token_path = Path(token_path_value)
            if not token_path.is_absolute():
                token_path = (Path.cwd() / token_path).resolve()
            token_ids = np.load(token_path, mmap_mode="r", allow_pickle=False)
            tokens = tokenizer.ids_to_tokens(token_ids)
            plans = plan_state_safe_chunks(tokens, context_length + 1)
            for plan in plans:
                reconstruction_ids = tuple(
                    tokenizer.token_to_id[token] for token in plan.reconstruction_tokens
                )
                chunks.append(
                    ChunkReference(
                        token_array_path=token_path,
                        source_file=_string_field(row, "source_file"),
                        tune_group=_string_field(row, "tune_group"),
                        split=split,
                        plan=plan,
                        reconstruction_ids=reconstruction_ids,
                    )
                )
                if maximum_chunks is not None and len(chunks) >= maximum_chunks:
                    break
            if maximum_chunks is not None and len(chunks) >= maximum_chunks:
                break
        if not chunks:
            raise ValueError(f"No trainable chunks found for split {split}")
        self.chunks = tuple(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.chunks[index]
        stored = np.load(reference.token_array_path, mmap_mode="r", allow_pickle=False)
        raw = np.asarray(
            stored[reference.plan.start : reference.plan.end], dtype=np.int64
        )
        if reference.reconstruction_ids:
            sequence = np.concatenate(
                [np.asarray(reference.reconstruction_ids, dtype=np.int64), raw]
            )
        else:
            sequence = raw
        if len(sequence) < 2:
            raise ValueError(f"Chunk is too short: {reference.source_file}")
        if len(sequence) > self.context_length + 1:
            raise ValueError(f"Chunk exceeds context: {reference.source_file}")
        tensor = torch.from_numpy(sequence.copy()).long()
        return tensor[:-1], tensor[1:]

    @property
    def performance_files(self) -> tuple[str, ...]:
        return tuple(sorted({chunk.source_file for chunk in self.chunks}))


def collate_token_chunks(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad inputs and mask padded targets for next-token loss."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    maximum_length = max(inputs.numel() for inputs, _ in batch)
    input_batch = torch.full(
        (len(batch), maximum_length), pad_token_id, dtype=torch.long
    )
    target_batch = torch.full((len(batch), maximum_length), -100, dtype=torch.long)
    for row, (inputs, targets) in enumerate(batch):
        length = inputs.numel()
        input_batch[row, :length] = inputs
        target_batch[row, :length] = targets
    return input_batch, target_batch


def default_pad_token_id(tokenizer: ExpressiveMidiTokenizer) -> int:
    """Use BOS as harmless right padding; padded targets are ignored."""
    return tokenizer.token_to_id[BOS_TOKEN]
