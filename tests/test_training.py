from __future__ import annotations

from pathlib import Path

import torch

from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.sampling import sample_tokens
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import (
    TrainingConfig,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
)


def small_model(vocabulary_size: int) -> PianoTransformer:
    return PianoTransformer(
        TransformerConfig(
            vocabulary_size=vocabulary_size,
            context_length=32,
            embedding_size=32,
            layer_count=1,
            head_count=4,
            feed_forward_size=64,
            dropout=0.0,
        )
    )


def small_training_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        maximum_steps=2,
        warmup_steps=0,
        evaluation_interval=1,
        evaluation_batches=1,
        checkpoint_interval=1,
        log_interval=1,
        amp=False,
    )


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path: Path) -> None:
    model = small_model(32)
    config = small_training_config()
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=3,
        best_validation_loss=1.25,
        model_config=model.config,
        training_config=config,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    loaded = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        map_location="cpu",
        restore_rng=True,
    )

    assert loaded["step"] == 3
    assert loaded["best_validation_loss"] == 1.25
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_sampling_is_deterministic_and_decodes_without_repairs() -> None:
    tokenizer = ExpressiveMidiTokenizer()
    torch.manual_seed(11)
    model = small_model(tokenizer.vocabulary_size).eval()

    first = sample_tokens(
        model,
        tokenizer,
        prompt_tokens=(BOS_TOKEN,),
        maximum_new_tokens=24,
        temperature=1.0,
        top_k=20,
        top_p=0.9,
        seed=99,
    )
    second = sample_tokens(
        model,
        tokenizer,
        prompt_tokens=(BOS_TOKEN,),
        maximum_new_tokens=24,
        temperature=1.0,
        top_k=20,
        top_p=0.9,
        seed=99,
    )

    assert first.tokens == second.tokens
    assert tokenizer.decode(first.tokens).repairs.total == 0
