from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.sampling import sample_tokens
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import (
    TrainingConfig,
    build_optimizer,
    evaluate_loss,
    load_checkpoint,
    load_model_checkpoint,
    save_checkpoint,
    train_model,
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

    inference_model, metadata = load_model_checkpoint(path, device=torch.device("cpu"))
    assert metadata["step"] == 3
    for name, value in inference_model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_evaluate_loss_is_token_weighted_and_preserves_mode() -> None:
    torch.manual_seed(17)
    model = small_model(32).eval()
    first = (torch.tensor([[1, 2]]), torch.tensor([[2, -100]]))
    second = (
        torch.tensor([[3, 4, 5, 6]]),
        torch.tensor([[4, 5, 6, 7]]),
    )
    with torch.inference_mode():
        _, first_loss = model(*first)
        _, second_loss = model(*second)
    assert first_loss is not None and second_loss is not None
    expected = (float(first_loss) + 4 * float(second_loss)) / 5

    observed = evaluate_loss(
        model,
        [first, second],
        device=torch.device("cpu"),
        maximum_batches=2,
        amp=False,
    )

    assert observed == pytest.approx(expected)
    assert model.training is False


def test_training_config_rejects_invalid_optimizer_settings() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        TrainingConfig(weight_decay=-0.1)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        TrainingConfig(gradient_clip_norm=0.0)
    with pytest.raises(ValueError, match="data_loader_workers"):
        TrainingConfig(data_loader_workers=-1)


def test_train_model_runs_one_step_and_writes_resumable_checkpoints(
    tmp_path: Path,
) -> None:
    model = small_model(32)
    config = small_training_config()
    batch = (
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
    )

    summary = train_model(
        model=model,
        train_loader=[batch],
        validation_loader=[batch],
        training_config=config,
        device=torch.device("cpu"),
        checkpoint_directory=tmp_path,
        maximum_steps=1,
    )

    assert summary["steps"] == 1
    assert summary["final_validation_loss"] > 0
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "latest.pt").is_file()


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
