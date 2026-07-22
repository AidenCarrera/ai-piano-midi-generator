"""Constrained temperature, top-k, and nucleus sampling for MIDI tokens."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import floor
from typing import Literal

import torch

from ai_piano_midi_generator.model import PianoTransformer
from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    PEDAL_UP_TOKEN,
    ExpressiveMidiTokenizer,
)


@dataclass(frozen=True, slots=True)
class SamplingResult:
    tokens: tuple[str, ...]
    invalid_events_blocked: dict[str, int]
    final_state_repairs: dict[str, int]
    safeguard_interventions: tuple[dict[str, int | float | str], ...] = ()


@dataclass(slots=True)
class _GenerationState:
    active_notes: dict[int, int]
    pedal_down: bool = False
    velocity_selected: bool = False
    ended: bool = False
    elapsed_steps: int = 0
    last_note_on_step: int | None = None


def sanitize_prompt_tokens(
    prompt_tokens: tuple[str, ...],
    tokenizer: ExpressiveMidiTokenizer,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Remove redundant/invalid prompt state events without changing source data."""
    state = _GenerationState(active_notes=defaultdict(int))
    repairs: Counter[str] = Counter()
    sanitized: list[str] = []
    for index, token in enumerate(prompt_tokens):
        if token not in tokenizer.token_to_id:
            raise ValueError(f"Unknown prompt token: {token}")
        if token == BOS_TOKEN:
            if index == 0 and not sanitized:
                sanitized.append(token)
            else:
                repairs["unexpected_bos"] += 1
            continue
        if token == EOS_TOKEN:
            repairs["removed_eos"] += 1
            break
        if token == PEDAL_DOWN_TOKEN and state.pedal_down:
            repairs["duplicate_pedal_down"] += 1
            continue
        if token == PEDAL_UP_TOKEN and not state.pedal_down:
            repairs["pedal_up_while_up"] += 1
            continue
        if token.startswith("NOTE_ON_") and not state.velocity_selected:
            repairs["note_on_without_velocity"] += 1
            continue
        if token.startswith("NOTE_OFF_"):
            pitch = int(token.removeprefix("NOTE_OFF_"))
            if state.active_notes.get(pitch, 0) <= 0:
                repairs["orphan_note_off"] += 1
                continue
        invalid: Counter[str] = Counter()
        _advance_state(state, token, invalid)
        if invalid:
            raise RuntimeError(f"Prompt sanitizer retained invalid event: {invalid}")
        sanitized.append(token)
    if not sanitized or sanitized[0] != BOS_TOKEN:
        sanitized.insert(0, BOS_TOKEN)
        repairs["inserted_bos"] += 1
    return tuple(sanitized), dict(sorted(repairs.items()))


def _advance_state(state: _GenerationState, token: str, invalid: Counter[str]) -> None:
    if token.startswith("TIME_SHIFT_"):
        state.elapsed_steps += int(token.removeprefix("TIME_SHIFT_"))
    elif token.startswith("VELOCITY_"):
        state.velocity_selected = True
    elif token.startswith("NOTE_ON_"):
        pitch = int(token.removeprefix("NOTE_ON_"))
        if not state.velocity_selected:
            invalid["note_on_without_velocity"] += 1
        state.active_notes[pitch] = state.active_notes.get(pitch, 0) + 1
        state.last_note_on_step = state.elapsed_steps
    elif token.startswith("NOTE_OFF_"):
        pitch = int(token.removeprefix("NOTE_OFF_"))
        if state.active_notes.get(pitch, 0) <= 0:
            invalid["orphan_note_off"] += 1
        else:
            state.active_notes[pitch] -= 1
    elif token == PEDAL_DOWN_TOKEN:
        if state.pedal_down:
            invalid["duplicate_pedal_down"] += 1
        state.pedal_down = True
    elif token == PEDAL_UP_TOKEN:
        if not state.pedal_down:
            invalid["pedal_up_while_up"] += 1
        state.pedal_down = False
    elif token == EOS_TOKEN:
        if any(state.active_notes.values()) or state.pedal_down:
            invalid["unsafe_eos"] += 1
        state.ended = True


def _constraint_mask(
    tokenizer: ExpressiveMidiTokenizer,
    state: _GenerationState,
    device: torch.device,
    *,
    allow_eos: bool,
) -> torch.Tensor:
    allowed = torch.ones(tokenizer.vocabulary_size, dtype=torch.bool, device=device)
    allowed[tokenizer.token_to_id[BOS_TOKEN]] = False
    if not state.velocity_selected:
        for pitch in range(128):
            allowed[tokenizer.token_to_id[f"NOTE_ON_{pitch}"]] = False
    for pitch in range(128):
        if state.active_notes.get(pitch, 0) <= 0:
            allowed[tokenizer.token_to_id[f"NOTE_OFF_{pitch}"]] = False
    if state.pedal_down:
        allowed[tokenizer.token_to_id[PEDAL_DOWN_TOKEN]] = False
    else:
        allowed[tokenizer.token_to_id[PEDAL_UP_TOKEN]] = False
    if any(state.active_notes.values()) or state.pedal_down:
        allowed[tokenizer.token_to_id[EOS_TOKEN]] = False
    if not allow_eos:
        allowed[tokenizer.token_to_id[EOS_TOKEN]] = False
    return allowed


def _blocked_reasons(
    token: str, state: _GenerationState, *, allow_eos: bool
) -> tuple[str, ...]:
    reasons: list[str] = []
    if token == BOS_TOKEN:
        reasons.append("unexpected_bos")
    elif token.startswith("NOTE_ON_") and not state.velocity_selected:
        reasons.append("note_on_without_velocity")
    elif token.startswith("NOTE_OFF_"):
        pitch = int(token.removeprefix("NOTE_OFF_"))
        if state.active_notes.get(pitch, 0) <= 0:
            reasons.append("orphan_note_off")
    elif token == PEDAL_DOWN_TOKEN and state.pedal_down:
        reasons.append("duplicate_pedal_down")
    elif token == PEDAL_UP_TOKEN and not state.pedal_down:
        reasons.append("pedal_up_while_up")
    elif token == EOS_TOKEN:
        if any(state.active_notes.values()) or state.pedal_down:
            reasons.append("unsafe_eos")
        if not allow_eos:
            reasons.append("minimum_length_eos")
    return tuple(reasons)


def _sample_index(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    filtered = logits / temperature
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        threshold = torch.topk(filtered, min(top_k, filtered.numel())).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(probabilities, dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            0, sorted_indices, sorted_logits
        )
    probabilities = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _seconds_to_steps(seconds: float | None, resolution: float) -> int | None:
    if seconds is None:
        return None
    if seconds <= 0:
        raise ValueError("Safeguard limits must be positive")
    return max(1, floor(seconds / resolution + 1e-9))


def _remaining_shift_steps(
    state: _GenerationState,
    *,
    duration_limit_steps: int | None,
    silence_limit_steps: int | None,
) -> tuple[int | None, str | None]:
    limits: list[tuple[int, str]] = []
    if duration_limit_steps is not None:
        limits.append((duration_limit_steps - state.elapsed_steps, "duration_limit"))
    if silence_limit_steps is not None:
        silence_start = state.last_note_on_step or 0
        limits.append(
            (
                silence_limit_steps - (state.elapsed_steps - silence_start),
                "silence_limit",
            )
        )
    if not limits:
        return None, None
    return min(limits, key=lambda item: item[0])


def _intervention(
    *,
    kind: str,
    action: str,
    generated_index: int,
    state: _GenerationState,
    proposed_shift_steps: int | None = None,
    masked_time_shift_tokens: int | None = None,
) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {
        "kind": kind,
        "action": action,
        "model_token_index": generated_index,
        "elapsed_steps": state.elapsed_steps,
    }
    if proposed_shift_steps is not None:
        result["proposed_shift_steps"] = proposed_shift_steps
    if masked_time_shift_tokens is not None:
        result["masked_time_shift_tokens"] = masked_time_shift_tokens
    return result


@torch.no_grad()
def sample_tokens(
    model: PianoTransformer,
    tokenizer: ExpressiveMidiTokenizer,
    *,
    prompt_tokens: tuple[str, ...] = (BOS_TOKEN,),
    maximum_new_tokens: int = 512,
    minimum_new_tokens: int = 0,
    temperature: float = 1.0,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    seed: int = 2026,
    constrain_states: bool = True,
    generation_mode: Literal["endurance", "musical"] = "musical",
    maximum_duration_seconds: float | None = None,
    maximum_silence_seconds: float | None = None,
    safeguard_action: Literal["eos", "mask_time_shifts"] = "eos",
) -> SamplingResult:
    """Sample tokens and optionally terminate on explicit timing safeguards."""
    if maximum_new_tokens <= 0:
        raise ValueError("maximum_new_tokens must be positive")
    if not 0 <= minimum_new_tokens <= maximum_new_tokens:
        raise ValueError(
            "minimum_new_tokens must be between zero and maximum_new_tokens"
        )
    if generation_mode not in ("endurance", "musical"):
        raise ValueError("generation_mode must be 'endurance' or 'musical'")
    if safeguard_action not in ("eos", "mask_time_shifts"):
        raise ValueError("Unknown safeguard_action")
    if generation_mode == "endurance" and (
        maximum_duration_seconds is not None or maximum_silence_seconds is not None
    ):
        raise ValueError("Endurance mode does not permit timing safeguards")
    if generation_mode == "endurance":
        minimum_new_tokens = maximum_new_tokens
    duration_limit_steps = _seconds_to_steps(
        maximum_duration_seconds, tokenizer.config.timing_resolution_seconds
    )
    silence_limit_steps = _seconds_to_steps(
        maximum_silence_seconds, tokenizer.config.timing_resolution_seconds
    )
    device = next(model.parameters()).device
    state = _GenerationState(active_notes=defaultdict(int))
    prompt_invalid: Counter[str] = Counter()
    for token in prompt_tokens:
        if token not in tokenizer.token_to_id:
            raise ValueError(f"Unknown prompt token: {token}")
        _advance_state(state, token, prompt_invalid)
    if prompt_invalid:
        raise ValueError(
            f"Prompt contains invalid state transitions: {dict(prompt_invalid)}"
        )
    generated = list(prompt_tokens)
    blocked: Counter[str] = Counter()
    interventions: list[dict[str, int | float | str]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.eval()

    for generated_index in range(maximum_new_tokens):
        if state.ended:
            break
        ids = torch.tensor(
            [tokenizer.token_to_id[token] for token in generated],
            dtype=torch.long,
            device=device,
        )[-model.config.context_length :]
        logits, _ = model(ids.unsqueeze(0))
        next_logits = logits[0, -1].float()
        if constrain_states:
            allow_eos = generated_index >= minimum_new_tokens
            allowed = _constraint_mask(
                tokenizer,
                state,
                device,
                allow_eos=allow_eos,
            )
            unconstrained_token = tokenizer.vocabulary[int(next_logits.argmax().item())]
            for reason in _blocked_reasons(
                unconstrained_token, state, allow_eos=allow_eos
            ):
                blocked[reason] += 1
            next_logits = next_logits.masked_fill(~allowed, -torch.inf)
        remaining_steps, limiting_kind = _remaining_shift_steps(
            state,
            duration_limit_steps=duration_limit_steps,
            silence_limit_steps=silence_limit_steps,
        )
        if remaining_steps is not None and remaining_steps <= 0:
            assert limiting_kind is not None
            if safeguard_action == "eos":
                interventions.append(
                    _intervention(
                        kind=limiting_kind,
                        action="terminate_with_eos",
                        generated_index=generated_index,
                        state=state,
                    )
                )
                break
        if remaining_steps is not None and safeguard_action == "mask_time_shifts":
            masked_count = 0
            for steps in range(
                max(1, remaining_steps + 1),
                tokenizer.config.maximum_time_shift_steps + 1,
            ):
                token_id = tokenizer.token_to_id[f"TIME_SHIFT_{steps}"]
                if torch.isfinite(next_logits[token_id]):
                    next_logits[token_id] = -torch.inf
                    masked_count += 1
            if masked_count:
                assert limiting_kind is not None
                interventions.append(
                    _intervention(
                        kind=limiting_kind,
                        action="mask_time_shifts",
                        generated_index=generated_index,
                        state=state,
                        masked_time_shift_tokens=masked_count,
                    )
                )
        next_id = _sample_index(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        token = tokenizer.vocabulary[next_id]
        proposed_shift = (
            int(token.removeprefix("TIME_SHIFT_"))
            if token.startswith("TIME_SHIFT_")
            else None
        )
        if (
            proposed_shift is not None
            and remaining_steps is not None
            and proposed_shift > remaining_steps
            and safeguard_action == "eos"
        ):
            assert limiting_kind is not None
            interventions.append(
                _intervention(
                    kind=limiting_kind,
                    action="terminate_with_eos",
                    generated_index=generated_index,
                    state=state,
                    proposed_shift_steps=proposed_shift,
                )
            )
            break
        generated.append(token)
        invalid: Counter[str] = Counter()
        _advance_state(state, token, invalid)
        if invalid and constrain_states:
            raise RuntimeError(f"Constrained sampler produced invalid token: {invalid}")

    repairs: Counter[str] = Counter()
    if not state.ended:
        for pitch in sorted(state.active_notes):
            for _ in range(state.active_notes[pitch]):
                generated.append(f"NOTE_OFF_{pitch}")
                repairs["closed_active_notes"] += 1
        if state.pedal_down:
            generated.append(PEDAL_UP_TOKEN)
            repairs["released_pedal"] += 1
        generated.append(EOS_TOKEN)
        repairs["appended_eos"] += 1
    return SamplingResult(
        tokens=tuple(generated),
        invalid_events_blocked=dict(sorted(blocked.items())),
        final_state_repairs=dict(sorted(repairs.items())),
        safeguard_interventions=tuple(interventions),
    )
