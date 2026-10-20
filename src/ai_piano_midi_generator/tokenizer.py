"""Expressive event tokenization for cleaned piano MIDI files."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from math import floor
from pathlib import Path
from typing import Literal

import numpy as np
from symusic.core import (
    ControlChangeTick,
    NoteTick,
    ScoreSecond,
    ScoreTick,
    TempoTick,
    TrackTick,
)

EventKind = Literal["note_off", "pedal_up", "pedal_down", "note_on"]

BOS_TOKEN = "BOS"
EOS_TOKEN = "EOS"
PEDAL_DOWN_TOKEN = "PEDAL_DOWN"
PEDAL_UP_TOKEN = "PEDAL_UP"


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Configurable quantization and fixed-tempo decoding settings."""

    timing_resolution_seconds: float = 0.01
    maximum_time_shift_seconds: float = 1.0
    velocity_bin_count: int = 32
    decode_tempo_qpm: float = 120.0
    decode_ticks_per_quarter: int = 480

    def __post_init__(self) -> None:
        if self.timing_resolution_seconds <= 0:
            raise ValueError("timing_resolution_seconds must be positive")
        if self.maximum_time_shift_seconds < self.timing_resolution_seconds:
            raise ValueError(
                "maximum_time_shift_seconds must be at least one timing step"
            )
        if not 1 <= self.velocity_bin_count <= 128:
            raise ValueError("velocity_bin_count must be between 1 and 128")
        if self.decode_tempo_qpm <= 0:
            raise ValueError("decode_tempo_qpm must be positive")
        if self.decode_ticks_per_quarter <= 0:
            raise ValueError("decode_ticks_per_quarter must be positive")

    @property
    def maximum_time_shift_steps(self) -> int:
        """Return the maximum integer step represented by one shift token."""
        return max(
            1,
            _round_half_up(
                self.maximum_time_shift_seconds / self.timing_resolution_seconds
            ),
        )

    @property
    def effective_maximum_time_shift_seconds(self) -> float:
        """Return the exact maximum shift represented after step conversion."""
        return self.maximum_time_shift_steps * self.timing_resolution_seconds

    def to_dict(self) -> dict[str, float | int]:
        """Return settings including the effective step-derived maximum."""
        result = asdict(self)
        result["maximum_time_shift_steps"] = self.maximum_time_shift_steps
        result["effective_maximum_time_shift_seconds"] = (
            self.effective_maximum_time_shift_seconds
        )
        return result


@dataclass(frozen=True, slots=True)
class EncodingDiagnostics:
    """Loss and event counts introduced by configurable quantization."""

    timing_errors_seconds: tuple[float, ...]
    velocity_errors: tuple[int, ...]
    event_counts: dict[str, int]
    minimum_duration_repairs: int = 0


@dataclass(frozen=True, slots=True)
class EncodedSequence:
    """Token sequence and its source-to-token diagnostics."""

    tokens: tuple[str, ...]
    diagnostics: EncodingDiagnostics


@dataclass(slots=True)
class DecodingRepairs:
    """Repairs used to turn arbitrary token input into valid MIDI."""

    missing_bos: int = 0
    missing_eos: int = 0
    unexpected_bos: int = 0
    tokens_after_eos: int = 0
    note_ons_without_velocity: int = 0
    orphan_note_offs: int = 0
    unclosed_notes: int = 0
    nonpositive_note_durations: int = 0

    @property
    def total(self) -> int:
        """Return the total number of repairs."""
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable repair mapping."""
        result = asdict(self)
        result["total"] = self.total
        return result


@dataclass(frozen=True, slots=True)
class DecodedSequence:
    """Fixed-tempo MIDI score and any repairs made while decoding."""

    score: ScoreTick
    repairs: DecodingRepairs


@dataclass(frozen=True, slots=True)
class _Event:
    step: int
    true_time: float
    kind: EventKind
    pitch: int | None = None
    velocity_bin: int | None = None
    velocity: int | None = None


def _round_half_up(value: float) -> int:
    return floor(value + 0.5)


class ExpressiveMidiTokenizer:
    """Encode expressive piano events and decode them at a fixed tempo."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.vocabulary = self._build_vocabulary()
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.vocabulary)
        }

    @property
    def vocabulary_size(self) -> int:
        """Return the number of possible tokens."""
        return len(self.vocabulary)

    def _build_vocabulary(self) -> tuple[str, ...]:
        return (
            BOS_TOKEN,
            EOS_TOKEN,
            PEDAL_DOWN_TOKEN,
            PEDAL_UP_TOKEN,
            *(f"NOTE_ON_{pitch}" for pitch in range(128)),
            *(f"NOTE_OFF_{pitch}" for pitch in range(128)),
            *(
                f"VELOCITY_{velocity_bin}"
                for velocity_bin in range(self.config.velocity_bin_count)
            ),
            *(
                f"TIME_SHIFT_{steps}"
                for steps in range(1, self.config.maximum_time_shift_steps + 1)
            ),
        )

    def velocity_to_bin(self, velocity: int) -> int:
        """Map a MIDI velocity to a configured bin."""
        if not 0 <= velocity <= 127:
            raise ValueError(f"MIDI velocity is out of range: {velocity}")
        return min(
            self.config.velocity_bin_count - 1,
            velocity * self.config.velocity_bin_count // 128,
        )

    def bin_to_velocity(self, velocity_bin: int) -> int:
        """Return the integer midpoint represented by a velocity bin."""
        if not 0 <= velocity_bin < self.config.velocity_bin_count:
            raise ValueError(f"Velocity bin is out of range: {velocity_bin}")
        lower = velocity_bin * 128 // self.config.velocity_bin_count
        upper = (velocity_bin + 1) * 128 // self.config.velocity_bin_count - 1
        return max(1, _round_half_up((lower + upper) / 2))

    def _time_to_step(self, time_seconds: float) -> int:
        return max(
            0,
            _round_half_up(time_seconds / self.config.timing_resolution_seconds),
        )

    def _step_to_tick(self, step: int) -> int:
        seconds = step * self.config.timing_resolution_seconds
        ticks_per_second = (
            self.config.decode_ticks_per_quarter * self.config.decode_tempo_qpm / 60.0
        )
        return _round_half_up(seconds * ticks_per_second)

    def _step_to_decoded_seconds(self, step: int) -> float:
        ticks_per_second = (
            self.config.decode_ticks_per_quarter * self.config.decode_tempo_qpm / 60.0
        )
        return self._step_to_tick(step) / ticks_per_second

    def _events(self, score: ScoreSecond) -> tuple[list[_Event], int]:
        events: list[_Event] = []
        minimum_duration_repairs = 0
        for track in score.tracks:
            for note in track.notes:
                start_time = note.time
                end_time = note.time + note.duration
                start_step = self._time_to_step(start_time)
                end_step = self._time_to_step(end_time)
                if end_step <= start_step:
                    end_step = start_step + 1
                    minimum_duration_repairs += 1
                velocity = note.velocity
                velocity_bin = self.velocity_to_bin(velocity)
                events.append(
                    _Event(
                        step=start_step,
                        true_time=start_time,
                        kind="note_on",
                        pitch=note.pitch,
                        velocity_bin=velocity_bin,
                        velocity=velocity,
                    )
                )
                events.append(
                    _Event(
                        step=end_step,
                        true_time=end_time,
                        kind="note_off",
                        pitch=note.pitch,
                    )
                )
            for control in track.controls:
                if control.number != 64:
                    continue
                time = control.time
                events.append(
                    _Event(
                        step=self._time_to_step(time),
                        true_time=time,
                        kind="pedal_down" if control.value >= 64 else "pedal_up",
                    )
                )
        return events, minimum_duration_repairs

    @staticmethod
    def _event_sort_key(event: _Event) -> tuple[int, int, int, int]:
        priorities = {
            "note_off": 0,
            "pedal_up": 1,
            "pedal_down": 2,
            "note_on": 3,
        }
        if event.kind == "note_on":
            return (
                event.step,
                priorities[event.kind],
                event.velocity_bin or 0,
                event.pitch or 0,
            )
        return (
            event.step,
            priorities[event.kind],
            event.pitch if event.pitch is not None else -1,
            0,
        )

    def encode(self, score: ScoreSecond) -> EncodedSequence:
        """Encode a seconds-domain score into deterministic event tokens."""
        events, minimum_duration_repairs = self._events(score)
        events.sort(key=self._event_sort_key)
        tokens = [BOS_TOKEN]
        current_step = 0
        current_event_step: int | None = None
        emitted_velocity_bin: int | None = None
        pedal_down = False
        timing_errors: list[float] = []
        velocity_errors: list[int] = []
        event_counts: Counter[str] = Counter()

        for event in events:
            if event.kind in ("pedal_down", "pedal_up"):
                # Binary sustain: CC64 values that keep the state are not events.
                # The sampler forbids these transitions, so training data must too.
                if (event.kind == "pedal_down") == pedal_down:
                    continue
                pedal_down = event.kind == "pedal_down"
            if event.step != current_event_step:
                delta = event.step - current_step
                while delta > 0:
                    shift = min(delta, self.config.maximum_time_shift_steps)
                    tokens.append(f"TIME_SHIFT_{shift}")
                    current_step += shift
                    delta -= shift
                current_event_step = event.step
                emitted_velocity_bin = None

            timing_errors.append(
                abs(event.true_time - self._step_to_decoded_seconds(event.step))
            )
            event_counts[event.kind] += 1
            if event.kind == "note_on":
                assert event.pitch is not None
                assert event.velocity_bin is not None
                assert event.velocity is not None
                if event.velocity_bin != emitted_velocity_bin:
                    tokens.append(f"VELOCITY_{event.velocity_bin}")
                    emitted_velocity_bin = event.velocity_bin
                tokens.append(f"NOTE_ON_{event.pitch}")
                velocity_errors.append(
                    abs(event.velocity - self.bin_to_velocity(event.velocity_bin))
                )
            elif event.kind == "note_off":
                assert event.pitch is not None
                tokens.append(f"NOTE_OFF_{event.pitch}")
            elif event.kind == "pedal_down":
                tokens.append(PEDAL_DOWN_TOKEN)
            else:
                tokens.append(PEDAL_UP_TOKEN)

        tokens.append(EOS_TOKEN)
        return EncodedSequence(
            tokens=tuple(tokens),
            diagnostics=EncodingDiagnostics(
                timing_errors_seconds=tuple(timing_errors),
                velocity_errors=tuple(velocity_errors),
                event_counts=dict(sorted(event_counts.items())),
                minimum_duration_repairs=minimum_duration_repairs,
            ),
        )

    def encode_file(self, path: Path) -> EncodedSequence:
        """Parse and encode a MIDI file in elapsed seconds."""
        return self.encode(ScoreSecond.from_file(path))

    def tokens_to_ids(self, tokens: tuple[str, ...] | list[str]) -> np.ndarray:
        """Convert tokens to the smallest supported compact integer array."""
        dtype = np.uint16 if self.vocabulary_size <= 65_536 else np.uint32
        try:
            return np.asarray(
                [self.token_to_id[token] for token in tokens], dtype=dtype
            )
        except KeyError as exc:
            raise ValueError(f"Unknown token: {exc.args[0]}") from exc

    def ids_to_tokens(self, token_ids: np.ndarray) -> tuple[str, ...]:
        """Convert a one-dimensional integer token array back to token strings."""
        if token_ids.ndim != 1:
            raise ValueError("Token ID array must be one-dimensional")
        result: list[str] = []
        for token_id_value in token_ids:
            token_id = int(token_id_value)
            if not 0 <= token_id < self.vocabulary_size:
                raise ValueError(f"Token ID is out of range: {token_id}")
            result.append(self.vocabulary[token_id])
        return tuple(result)

    def decode(self, tokens: tuple[str, ...] | list[str]) -> DecodedSequence:
        """Decode tokens into a valid single-track, fixed-tempo MIDI score."""
        repairs = DecodingRepairs()
        if not tokens or tokens[0] != BOS_TOKEN:
            repairs.missing_bos += 1

        current_step = 0
        current_velocity_bin: int | None = None
        active_notes: dict[int, deque[tuple[int, int]]] = defaultdict(deque)
        completed_notes: list[tuple[int, int, int, int]] = []
        pedal_events: list[tuple[int, int]] = []
        saw_eos = False

        for index, token in enumerate(tokens):
            if saw_eos:
                repairs.tokens_after_eos += 1
                continue
            if token == BOS_TOKEN:
                if index != 0:
                    repairs.unexpected_bos += 1
                continue
            if token == EOS_TOKEN:
                saw_eos = True
                continue
            if token.startswith("TIME_SHIFT_"):
                shift = self._parse_index_token(token, "TIME_SHIFT_")
                if not 1 <= shift <= self.config.maximum_time_shift_steps:
                    raise ValueError(f"Time-shift token is out of range: {token}")
                current_step += shift
                continue
            if token.startswith("VELOCITY_"):
                velocity_bin = self._parse_index_token(token, "VELOCITY_")
                if not 0 <= velocity_bin < self.config.velocity_bin_count:
                    raise ValueError(f"Velocity token is out of range: {token}")
                current_velocity_bin = velocity_bin
                continue
            if token.startswith("NOTE_ON_"):
                pitch = self._parse_pitch_token(token, "NOTE_ON_")
                if current_velocity_bin is None:
                    current_velocity_bin = self.velocity_to_bin(64)
                    repairs.note_ons_without_velocity += 1
                active_notes[pitch].append(
                    (current_step, self.bin_to_velocity(current_velocity_bin))
                )
                continue
            if token.startswith("NOTE_OFF_"):
                pitch = self._parse_pitch_token(token, "NOTE_OFF_")
                if not active_notes[pitch]:
                    repairs.orphan_note_offs += 1
                    continue
                start_step, velocity = active_notes[pitch].popleft()
                end_step = current_step
                if end_step <= start_step:
                    end_step = start_step + 1
                    repairs.nonpositive_note_durations += 1
                completed_notes.append((start_step, end_step, pitch, velocity))
                continue
            if token == PEDAL_DOWN_TOKEN:
                pedal_events.append((current_step, 127))
                continue
            if token == PEDAL_UP_TOKEN:
                pedal_events.append((current_step, 0))
                continue
            raise ValueError(f"Unknown token: {token}")

        if not saw_eos:
            repairs.missing_eos += 1
        closing_step = current_step + 1
        for pitch, starts in active_notes.items():
            while starts:
                start_step, velocity = starts.popleft()
                end_step = max(closing_step, start_step + 1)
                completed_notes.append((start_step, end_step, pitch, velocity))
                repairs.unclosed_notes += 1

        score = ScoreTick(self.config.decode_ticks_per_quarter)
        score.tempos.append(TempoTick(0, self.config.decode_tempo_qpm))
        track = TrackTick("Tokenized piano", 0, False)
        for start_step, end_step, pitch, velocity in sorted(completed_notes):
            start_tick = self._step_to_tick(start_step)
            end_tick = self._step_to_tick(end_step)
            if end_tick <= start_tick:
                end_tick = start_tick + 1
                repairs.nonpositive_note_durations += 1
            track.notes.append(
                NoteTick(start_tick, end_tick - start_tick, pitch, velocity)
            )
        for step, value in pedal_events:
            track.controls.append(
                ControlChangeTick(self._step_to_tick(step), 64, value)
            )
        track.notes.sort()
        track.controls.sort()
        score.tracks.append(track)
        return DecodedSequence(score=score, repairs=repairs)

    @staticmethod
    def _parse_index_token(token: str, prefix: str) -> int:
        try:
            return int(token.removeprefix(prefix))
        except ValueError as exc:
            raise ValueError(f"Malformed token: {token}") from exc

    @classmethod
    def _parse_pitch_token(cls, token: str, prefix: str) -> int:
        pitch = cls._parse_index_token(token, prefix)
        if not 0 <= pitch <= 127:
            raise ValueError(f"Pitch token is out of range: {token}")
        return pitch
