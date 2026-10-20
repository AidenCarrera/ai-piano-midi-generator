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
from ai_piano_midi_generator.sampling import sample_tokens, sanitize_prompt_tokens
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
    test_loss = result["test_loss"]
    assert isinstance(test_loss, (int, float))
    assert test_loss > 0
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


class _AlwaysShiftModel(nn.Module):
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
        logits[..., self.tokenizer.token_to_id["TIME_SHIFT_1"]] = 100.0
        return logits, None


class _RecordingShiftModel(_AlwaysShiftModel):
    def __init__(self, tokenizer: ExpressiveMidiTokenizer) -> None:
        super().__init__(tokenizer)
        self.inputs: list[tuple[str, ...]] = []

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.inputs.append(self.tokenizer.ids_to_tokens(input_ids[0].cpu().numpy()))
        return super().forward(input_ids)


def test_long_generation_keeps_training_chunk_layout_in_model_window() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = _RecordingShiftModel(tokenizer)
    prompt = (BOS_TOKEN, PEDAL_DOWN_TOKEN, "VELOCITY_8", "NOTE_ON_60")

    sample_tokens(
        cast(PianoTransformer, model),
        tokenizer,
        prompt_tokens=prompt,
        maximum_new_tokens=30,
        top_k=1,
        top_p=1.0,
    )

    assert len(model.inputs) == 30
    assert all(len(window) <= 16 for window in model.inputs)
    assert all(window[0] == BOS_TOKEN for window in model.inputs)
    # Once the window slides, held note and pedal state are rebuilt explicitly.
    assert model.inputs[-1][:4] == prompt
    assert model.inputs[-1][4:] == ("TIME_SHIFT_1",) * 12


class _PrefersNoteOffModel(_AlwaysShiftModel):
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, _ = super().forward(input_ids)
        logits[..., self.tokenizer.token_to_id["NOTE_OFF_60"]] = 200.0
        return logits, None


def test_note_off_cannot_close_a_note_started_in_the_same_step() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _PrefersNoteOffModel(tokenizer))

    result = sample_tokens(
        model,
        tokenizer,
        prompt_tokens=(BOS_TOKEN, "VELOCITY_8", "NOTE_ON_60"),
        maximum_new_tokens=2,
        top_k=1,
        top_p=1.0,
    )

    assert result.tokens[3:5] == ("TIME_SHIFT_1", "NOTE_OFF_60")
    assert result.invalid_events_blocked == {"zero_duration_note_off": 1}
    assert tokenizer.decode(result.tokens).repairs.total == 0


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


def test_eos_is_allowed_from_the_first_prediction_without_a_minimum() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _AlwaysEosModel(tokenizer))

    result = sample_tokens(
        model,
        tokenizer,
        maximum_new_tokens=2048,
        minimum_new_tokens=0,
        top_k=1,
        top_p=1.0,
        seed=7,
    )

    assert result.tokens == (BOS_TOKEN, EOS_TOKEN)
    assert result.final_state_repairs == {}
    assert result.safeguard_interventions == ()


def test_model_eos_closes_active_state_only_in_exported_tokens() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _AlwaysEosModel(tokenizer))
    prompt = (
        BOS_TOKEN,
        PEDAL_DOWN_TOKEN,
        "VELOCITY_8",
        "NOTE_ON_60",
    )

    result = sample_tokens(
        model,
        tokenizer,
        prompt_tokens=prompt,
        maximum_new_tokens=10,
        top_k=1,
        top_p=1.0,
    )

    assert result.tokens == (
        *prompt,
        "TIME_SHIFT_1",
        "NOTE_OFF_60",
        "PEDAL_UP",
        EOS_TOKEN,
    )
    assert result.model_generated_token_count == 1
    assert result.model_eos_token_position == 1
    assert result.final_state_repairs == {
        "advanced_zero_duration_notes": 1,
        "closed_active_notes": 1,
        "released_pedal": 1,
    }
    assert tokenizer.decode(result.tokens).repairs.total == 0


def test_prompt_sanitizer_removes_redundant_pedal_state_events() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    prompt = (
        BOS_TOKEN,
        "PEDAL_UP",
        PEDAL_DOWN_TOKEN,
        PEDAL_DOWN_TOKEN,
        "PEDAL_UP",
        "PEDAL_UP",
    )

    sanitized, repairs = sanitize_prompt_tokens(prompt, tokenizer)

    assert sanitized == (BOS_TOKEN, PEDAL_DOWN_TOKEN, "PEDAL_UP")
    assert repairs == {"duplicate_pedal_down": 1, "pedal_up_while_up": 2}


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


def test_duration_limit_terminates_without_forcing_a_note() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _AlwaysShiftModel(tokenizer))

    result = sample_tokens(
        model,
        tokenizer,
        maximum_new_tokens=20,
        minimum_new_tokens=10,
        top_k=1,
        top_p=1.0,
        maximum_duration_seconds=0.03,
    )

    assert result.tokens == (
        BOS_TOKEN,
        "TIME_SHIFT_1",
        "TIME_SHIFT_1",
        "TIME_SHIFT_1",
        EOS_TOKEN,
    )
    assert result.safeguard_interventions[0]["kind"] == "duration_limit"
    assert result.safeguard_interventions[0]["action"] == "terminate_with_eos"


def test_silence_limit_tracks_time_since_last_note_on_and_preserves_grammar() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _AlwaysShiftModel(tokenizer))
    prompt = (BOS_TOKEN, "VELOCITY_8", "NOTE_ON_60")

    result = sample_tokens(
        model,
        tokenizer,
        prompt_tokens=prompt,
        maximum_new_tokens=20,
        top_k=1,
        top_p=1.0,
        maximum_silence_seconds=0.02,
    )
    decoded = tokenizer.decode(result.tokens)

    assert result.safeguard_interventions[0]["kind"] == "silence_limit"
    assert result.final_state_repairs == {
        "appended_eos": 1,
        "closed_active_notes": 1,
    }
    assert decoded.repairs.total == 0


def test_safeguards_are_disabled_by_default() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _AlwaysShiftModel(tokenizer))

    result = sample_tokens(
        model,
        tokenizer,
        maximum_new_tokens=7,
        top_k=1,
        top_p=1.0,
    )

    assert result.tokens.count("TIME_SHIFT_1") == 7
    assert result.safeguard_interventions == ()


def test_endurance_mode_forces_exact_model_token_count_without_safeguards() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = cast(PianoTransformer, _AlwaysEosModel(tokenizer))

    result = sample_tokens(
        model,
        tokenizer,
        maximum_new_tokens=7,
        minimum_new_tokens=0,
        top_k=1,
        top_p=1.0,
        generation_mode="endurance",
    )

    assert result.tokens.count("TIME_SHIFT_1") == 7
    assert result.tokens[-1] == EOS_TOKEN
    assert result.safeguard_interventions == ()


def test_endurance_mode_rejects_safeguards() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    model = tiny_model(tokenizer.vocabulary_size)

    with pytest.raises(ValueError, match="Endurance mode"):
        sample_tokens(
            model,
            tokenizer,
            generation_mode="endurance",
            maximum_duration_seconds=10.0,
        )
