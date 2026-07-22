"""Plain-PyTorch causal decoder-only Transformer baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Architecture settings for the decoder-only baseline."""

    vocabulary_size: int = 392
    context_length: int = 2048
    embedding_size: int = 384
    layer_count: int = 6
    head_count: int = 6
    feed_forward_size: int = 1536
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0 or self.context_length <= 0:
            raise ValueError("Vocabulary and context sizes must be positive")
        if self.embedding_size <= 0 or self.layer_count <= 0:
            raise ValueError("Embedding size and layer count must be positive")
        if self.embedding_size % self.head_count:
            raise ValueError("embedding_size must be divisible by head_count")
        if self.feed_forward_size < self.embedding_size:
            raise ValueError("feed_forward_size must be at least embedding_size")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention using PyTorch SDPA."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_count = config.head_count
        self.head_size = config.embedding_size // config.head_count
        self.dropout = config.dropout
        self.query_key_value = nn.Linear(
            config.embedding_size, 3 * config.embedding_size
        )
        self.output = nn.Linear(config.embedding_size, config.embedding_size)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = hidden.shape
        query, key, value = self.query_key_value(hidden).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.head_count, self.head_size
            ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            heads(query),
            heads(key),
            heads(value),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, embedding_size)
        )
        return self.residual_dropout(self.output(attended))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_size, config.feed_forward_size),
            nn.GELU(),
            nn.Linear(config.feed_forward_size, config.embedding_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden)


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class PianoTransformer(nn.Module):
    """Causal language model over expressive MIDI event tokens."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.embedding_size
        )
        self.position_embedding = nn.Embedding(
            config.context_length, config.embedding_size
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.layer_count)
        )
        self.final_norm = nn.LayerNorm(config.embedding_size)
        self.output = nn.Linear(
            config.embedding_size, config.vocabulary_size, bias=False
        )
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context "
                f"{self.config.context_length}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def parameter_count(self) -> int:
        """Return the number of unique trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters())
