"""Constrained temperature, top-k, and nucleus sampling for MIDI tokens."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

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


@dataclass(slots=True)
class _GenerationState:
    active_notes: dict[int, int]
    pedal_down: bool = False
    velocity_selected: bool = False
    ended: bool = False


def _advance_state(state: _GenerationState, token: str, invalid: Counter[str]) -> None:
    if token.startswith("VELOCITY_"):
        state.velocity_selected = True
    elif token.startswith("NOTE_ON_"):
        pitch = int(token.removeprefix("NOTE_ON_"))
        if not state.velocity_selected:
            invalid["note_on_without_velocity"] += 1
        state.active_notes[pitch] = state.active_notes.get(pitch, 0) + 1
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
    blocked: Counter[str],
    device: torch.device,
) -> torch.Tensor:
    allowed = torch.ones(tokenizer.vocabulary_size, dtype=torch.bool, device=device)
    allowed[tokenizer.token_to_id[BOS_TOKEN]] = False
    if not state.velocity_selected:
        for pitch in range(128):
            allowed[tokenizer.token_to_id[f"NOTE_ON_{pitch}"]] = False
        blocked["note_on_without_velocity"] += 1
    for pitch in range(128):
        if state.active_notes.get(pitch, 0) <= 0:
            allowed[tokenizer.token_to_id[f"NOTE_OFF_{pitch}"]] = False
    if state.pedal_down:
        allowed[tokenizer.token_to_id[PEDAL_DOWN_TOKEN]] = False
        blocked["duplicate_pedal_down"] += 1
    else:
        allowed[tokenizer.token_to_id[PEDAL_UP_TOKEN]] = False
        blocked["pedal_up_while_up"] += 1
    if any(state.active_notes.values()) or state.pedal_down:
        allowed[tokenizer.token_to_id[EOS_TOKEN]] = False
        blocked["unsafe_eos"] += 1
    return allowed


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


@torch.no_grad()
def sample_tokens(
    model: PianoTransformer,
    tokenizer: ExpressiveMidiTokenizer,
    *,
    prompt_tokens: tuple[str, ...] = (BOS_TOKEN,),
    maximum_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    seed: int = 2026,
    constrain_states: bool = True,
) -> SamplingResult:
    """Sample tokens and close active musical state before returning."""
    if maximum_new_tokens <= 0:
        raise ValueError("maximum_new_tokens must be positive")
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
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.eval()

    for _ in range(maximum_new_tokens):
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
            allowed = _constraint_mask(tokenizer, state, blocked, device)
            next_logits = next_logits.masked_fill(~allowed, -torch.inf)
        next_id = _sample_index(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        token = tokenizer.vocabulary[next_id]
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
    )
