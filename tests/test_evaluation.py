from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from ai_piano_midi_generator.evaluation import (
    TOKEN_CATEGORIES,
    evaluate_test_model,
    invalid_prediction_tendencies,
    token_category,
)
from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.sampling import sample_tokens
from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    ExpressiveMidiTokenizer,
)


def tiny_model(vocabulary_size: int) -> PianoTransformer:
    return PianoTransformer(
        TransformerConfig(
            vocabulary_size=vocabulary_size,
            context_length=16,
            embedding_size=16,
            layer_count=1,
            head_count=2,
            feed_forward_size=32,
            dropout=0.0,
        )
    )


def test_every_vocabulary_token_has_an_evaluation_category() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    categories = {token_category(token) for token in tokenizer.vocabulary}

    assert categories == set(TOKEN_CATEGORIES)


def test_test_metrics_count_categories_and_invalid_predictions() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = tiny_model(tokenizer.vocabulary_size)
    tokens = (
        BOS_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_60",
        "TIME_SHIFT_10",
        "NOTE_OFF_60",
        PEDAL_DOWN_TOKEN,
        "PEDAL_UP",
        EOS_TOKEN,
    )
    ids = torch.tensor(tokenizer.tokens_to_ids(tokens).astype("int64"))

    def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yield ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)

    result = evaluate_test_model(
        model, batches(), tokenizer, device=torch.device("cpu"), amp=False
    )

    assert result["evaluated_token_count"] == len(tokens) - 1
    assert result["parameter_count"] == model.parameter_count()
    assert float(result["test_loss"]) > 0
    categories = result["token_accuracy_by_category"]
    assert isinstance(categories, dict)
    assert sum(int(value["count"]) for value in categories.values()) == len(tokens) - 1


def test_invalid_tendencies_use_observed_prefix_state() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    observed = (BOS_TOKEN, PEDAL_DOWN_TOKEN, "VELOCITY_8", "NOTE_ON_60")
    predicted = (
        "NOTE_ON_62",
        PEDAL_DOWN_TOKEN,
        "NOTE_OFF_61",
        EOS_TOKEN,
    )
    inputs = torch.tensor(tokenizer.tokens_to_ids(observed).astype("int64")).unsqueeze(
        0
    )
    predictions = torch.tensor(
        tokenizer.tokens_to_ids(predicted).astype("int64")
    ).unsqueeze(0)
    valid = torch.ones_like(inputs, dtype=torch.bool)

    tendencies = invalid_prediction_tendencies(inputs, predictions, valid, tokenizer)

    assert tendencies["note_on_without_velocity"] == 1
    assert tendencies["duplicate_pedal_down"] == 1
    assert tendencies["orphan_note_off"] == 1
    assert tendencies["unsafe_eos"] == 1
    assert tendencies["unclosed_notes"] == 1


class _AlwaysEosModel(nn.Module):
    def __init__(self, tokenizer: ExpressiveMidiTokenizer) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(context_length=16)
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits = torch.full(
            (*input_ids.shape, self.tokenizer.vocabulary_size),
            -100.0,
            device=input_ids.device,
        )
        logits[..., self.tokenizer.token_to_id[EOS_TOKEN]] = 100.0
        logits[..., self.tokenizer.token_to_id["TIME_SHIFT_1"]] = 0.0
        return logits, None


def test_minimum_generation_length_blocks_early_eos() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _AlwaysEosModel(tokenizer))

    result = sample_tokens(
        model,
        tokenizer,
        maximum_new_tokens=10,
        minimum_new_tokens=5,
        top_k=1,
        top_p=1.0,
        seed=7,
    )

    assert result.tokens[-1] == EOS_TOKEN
    assert result.tokens.index(EOS_TOKEN) >= 6
    assert result.invalid_events_blocked["minimum_length_eos"] == 5


@pytest.mark.parametrize("minimum", [-1, 11])
def test_invalid_minimum_generation_length_is_rejected(minimum: int) -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = tiny_model(tokenizer.vocabulary_size)

    with pytest.raises(ValueError, match="minimum_new_tokens"):
        sample_tokens(
            model,
            tokenizer,
            maximum_new_tokens=10,
            minimum_new_tokens=minimum,
        )
