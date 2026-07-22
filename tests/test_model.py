from __future__ import annotations

import pytest
import torch

from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=32,
        context_length=16,
        embedding_size=32,
        layer_count=2,
        head_count=4,
        feed_forward_size=64,
        dropout=0.0,
    )


def test_config_rejects_nonpositive_head_count() -> None:
    with pytest.raises(ValueError, match="head count"):
        TransformerConfig(head_count=0)


def test_model_output_shapes_and_shifted_loss() -> None:
    model = PianoTransformer(tiny_config())
    inputs = torch.randint(0, 32, (2, 12))
    targets = torch.randint(0, 32, (2, 12))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 12, 32)
    assert loss is not None
    assert loss.ndim == 0


def test_causal_attention_prevents_future_token_leakage() -> None:
    torch.manual_seed(7)
    model = PianoTransformer(tiny_config()).eval()
    first = torch.randint(0, 32, (1, 10))
    second = first.clone()
    second[0, -1] = (second[0, -1] + 1) % 32

    first_logits, _ = model(first)
    second_logits, _ = model(second)

    assert torch.allclose(first_logits[:, :-1], second_logits[:, :-1], atol=1e-6)
    assert not torch.allclose(first_logits[:, -1], second_logits[:, -1])


def test_one_forward_backward_pass_produces_gradients() -> None:
    model = PianoTransformer(tiny_config())
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
