from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import ai_piano_midi_generator.training as training_module
from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.sampling import sample_tokens
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import (
    TrainingConfig,
    build_optimizer,
    configure_determinism,
    evaluate_loss,
    load_checkpoint,
    load_model_checkpoint,
    save_checkpoint,
    set_random_seed,
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

    cpu_optimizer = build_optimizer(small_model(32), TrainingConfig())
    assert cpu_optimizer.defaults["fused"] is False


def test_training_config_rejects_unknown_compile_mode() -> None:
    with pytest.raises(ValueError, match="compile_mode"):
        TrainingConfig(compile_mode="fastest")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_build_optimizer_uses_fused_adamw_on_cuda() -> None:
    model = small_model(32).cuda()
    optimizer = build_optimizer(model, TrainingConfig(fused_adamw=True))
    assert optimizer.defaults["fused"] is True

    inputs = torch.tensor([[1, 2, 3, 4]], device="cuda")
    targets = torch.tensor([[2, 3, 4, 5]], device="cuda")
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()


def test_seed_and_determinism_are_configured_independently() -> None:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        set_random_seed(42)
        first = torch.rand(4)
        set_random_seed(42)
        assert torch.equal(torch.rand(4), first)

        configure_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark

        configure_determinism(False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.use_deterministic_algorithms(previous_deterministic)


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
        history_path=tmp_path / "history.jsonl",
    )

    assert summary["steps"] == 1
    assert isinstance(summary["final_validation_loss"], float)
    assert summary["final_validation_loss"] > 0
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "latest.pt").is_file()
    history = [
        json.loads(line)
        for line in (tmp_path / "history.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in history] == [1, 1]
    assert "training_loss" in history[0]
    assert history[1]["validation_loss"] == summary["final_validation_loss"]


def test_train_model_synchronizes_only_at_timing_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = small_model(32)
    config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        maximum_steps=2,
        warmup_steps=0,
        evaluation_interval=2,
        checkpoint_interval=2,
        log_interval=2,
        amp=False,
    )
    synchronization_calls: list[torch.device] = []
    monkeypatch.setattr(
        training_module,
        "_synchronize_device",
        synchronization_calls.append,
    )
    batch = (
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
    )

    train_model(
        model=model,
        train_loader=[batch],
        validation_loader=[batch],
        training_config=config,
        device=torch.device("cpu"),
        checkpoint_directory=tmp_path,
    )

    assert synchronization_calls == [
        torch.device("cpu"),
        torch.device("cpu"),
        torch.device("cpu"),
    ]


def test_train_model_compiles_in_place_with_configured_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = small_model(32)
    config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        maximum_steps=1,
        warmup_steps=0,
        evaluation_interval=1,
        checkpoint_interval=1,
        log_interval=1,
        amp=False,
        compile_model=True,
        compile_mode="max-autotune-no-cudagraphs",
    )
    compile_calls: list[dict[str, object]] = []

    def record_compile(self: PianoTransformer, *args: object, **kwargs: object) -> None:
        assert self is model
        assert not args
        compile_calls.append(kwargs)

    monkeypatch.setattr(PianoTransformer, "compile", record_compile)
    batch = (
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
    )

    train_model(
        model=model,
        train_loader=[batch],
        validation_loader=[batch],
        training_config=config,
        device=torch.device("cpu"),
        checkpoint_directory=tmp_path,
        maximum_steps=1,
    )

    assert compile_calls == [{"dynamic": False, "mode": "max-autotune-no-cudagraphs"}]


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
