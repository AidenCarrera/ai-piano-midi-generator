"""Split-aware token chunk loading with explicit musical state reconstruction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ai_piano_midi_generator.data.preparation import (
    ChunkPlan,
    plan_state_safe_chunks,
    plan_tail_excerpts,
)
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
    is_tail_excerpt: bool = False


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
        context_length: int = 2048,
        maximum_performances: int | None = None,
        maximum_chunks: int | None = None,
        tail_excerpt_lengths: Sequence[int] = (),
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown dataset split: {split}")
        if context_length < 8:
            raise ValueError("context_length must be at least 8")
        if tail_excerpt_lengths and split != "train":
            raise ValueError("Tail excerpts are only permitted for the training split")
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

        standard_chunks: list[ChunkReference] = []
        tail_chunks: list[ChunkReference] = []
        token_arrays: dict[Path, np.ndarray] = {}
        for row in rows:
            token_path_value = _string_field(row, "token_array_path")
            token_path = Path(token_path_value)
            if not token_path.is_absolute():
                token_path = (Path.cwd() / token_path).resolve()
            token_ids = token_arrays.get(token_path)
            if token_ids is None:
                token_ids = np.load(token_path, allow_pickle=False)
                if token_ids.ndim != 1:
                    raise ValueError(
                        f"Token array must be one-dimensional: {token_path}"
                    )
                token_ids.setflags(write=False)
                token_arrays[token_path] = token_ids
            tokens = tokenizer.ids_to_tokens(token_ids)
            plans = plan_state_safe_chunks(tokens, context_length + 1)
            for plan in plans:
                reconstruction_ids = tuple(
                    tokenizer.token_to_id[token] for token in plan.reconstruction_tokens
                )
                standard_chunks.append(
                    ChunkReference(
                        token_array_path=token_path,
                        source_file=_string_field(row, "source_file"),
                        tune_group=_string_field(row, "tune_group"),
                        split=split,
                        plan=plan,
                        reconstruction_ids=reconstruction_ids,
                    )
                )
                if (
                    maximum_chunks is not None
                    and len(standard_chunks) >= maximum_chunks
                ):
                    break
            if tail_excerpt_lengths:
                for plan in plan_tail_excerpts(
                    tokens,
                    tail_excerpt_lengths,
                    maximum_length=context_length + 1,
                ):
                    tail_chunks.append(
                        ChunkReference(
                            token_array_path=token_path,
                            source_file=_string_field(row, "source_file"),
                            tune_group=_string_field(row, "tune_group"),
                            split=split,
                            plan=plan,
                            reconstruction_ids=tuple(
                                tokenizer.token_to_id[token]
                                for token in plan.reconstruction_tokens
                            ),
                            is_tail_excerpt=True,
                        )
                    )
            if maximum_chunks is not None and len(standard_chunks) >= maximum_chunks:
                break
        if not standard_chunks:
            raise ValueError(f"No trainable chunks found for split {split}")
        self.standard_chunk_count = len(standard_chunks)
        self.tail_excerpt_count = len(tail_chunks)
        self.chunks = tuple([*standard_chunks, *tail_chunks])
        self._token_arrays = token_arrays

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.chunks[index]
        stored = self._token_arrays[reference.token_array_path]
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

    @property
    def cached_token_bytes(self) -> int:
        """Return the in-memory size of the split's unique token arrays."""
        return sum(array.nbytes for array in self._token_arrays.values())

    def mixed_sampler(self, *, tail_ratio: float, seed: int) -> WeightedRandomSampler:
        """Sample a fixed-size epoch with the requested standard/tail mass."""
        if not 0.0 < tail_ratio < 1.0:
            raise ValueError("tail_ratio must be between zero and one")
        if self.tail_excerpt_count == 0:
            raise ValueError("Cannot mix tails because the dataset has none")
        standard_weight = (1.0 - tail_ratio) / self.standard_chunk_count
        tail_weight = tail_ratio / self.tail_excerpt_count
        weights = [
            *([standard_weight] * self.standard_chunk_count),
            *([tail_weight] * self.tail_excerpt_count),
        ]
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            weights,
            num_samples=self.standard_chunk_count,
            replacement=True,
            generator=generator,
        )


def collate_token_chunks(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int = 0,
    padded_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad inputs and mask padded targets for next-token loss."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    maximum_length = max(inputs.numel() for inputs, _ in batch)
    if padded_length is not None:
        if padded_length < maximum_length:
            raise ValueError(
                f"padded_length {padded_length} is shorter than batch maximum "
                f"{maximum_length}"
            )
        maximum_length = padded_length
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
