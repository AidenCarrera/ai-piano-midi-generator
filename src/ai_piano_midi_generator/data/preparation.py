"""Grouped splitting and state-aware tokenized dataset preparation."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

import numpy as np

from ai_piano_midi_generator.tokenizer import (
    BOS_TOKEN,
    PEDAL_DOWN_TOKEN,
    PEDAL_UP_TOKEN,
    ExpressiveMidiTokenizer,
    TokenizerConfig,
)

DatasetSplit = Literal["train", "validation", "test"]
SPLITS: tuple[DatasetSplit, ...] = ("train", "validation", "test")

_GROUP_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_DESCRIPTOR_WORDS = frozenset(
    {
        "arrangement",
        "arranged",
        "barron",
        "bill",
        "chords",
        "doug",
        "edited",
        "elf",
        "evans",
        "file",
        "gm",
        "jazz",
        "kit",
        "latin",
        "mckenzie",
        "midi",
        "piano",
        "reharmonized",
        "solo",
        "trio",
        "version",
        "versions",
        "xg",
    }
)
_JOINED_SUFFIXES = (
    "reharmonized",
    "arrangement",
    "versions",
    "version",
    "chords",
    "piano",
    "trio",
    "solo",
    "midi",
    "latin",
    "gm",
    "xg",
)


@dataclass(frozen=True, slots=True)
class TuneGroupOverride:
    """Versionable manual mapping from one performance to a tune group."""

    relative_path: str
    tune_group: str


@dataclass(frozen=True, slots=True)
class GroupWeight:
    """Weight used for deterministic group-level split balancing."""

    tune_group: str
    file_count: int
    token_count: int


@dataclass(frozen=True, slots=True)
class TokenState:
    """Performance state immediately before one token-array position."""

    active_notes: tuple[tuple[int, int], ...]
    pedal_down: bool
    velocity_bin: int | None

    @property
    def safe(self) -> bool:
        """Return whether a fresh chunk can begin without musical state."""
        return not self.active_notes and not self.pedal_down


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """One raw token span plus an optional explicit state prefix."""

    start: int
    end: int
    context_length: int
    safe_start: bool
    state_reconstruction_required: bool
    active_notes_at_start: int
    pedal_down_at_start: bool
    reconstruction_tokens: tuple[str, ...]
    forced_inside_event_group: bool = False

    @property
    def raw_token_count(self) -> int:
        return self.end - self.start

    @property
    def model_token_count(self) -> int:
        return self.raw_token_count + len(self.reconstruction_tokens)


def _normalized_relative_path(value: str) -> str:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def load_tune_group_overrides(path: Path) -> dict[str, TuneGroupOverride]:
    """Load and validate a JSONL manual tune-group override manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Tune-group override manifest not found: {path}")
    overrides: dict[str, TuneGroupOverride] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object in {path} line {line_number}")
        relative_path = row.get("relative_path")
        tune_group = row.get("tune_group")
        if not isinstance(relative_path, str) or not isinstance(tune_group, str):
            raise ValueError(
                f"relative_path and tune_group must be strings in {path} "
                f"line {line_number}"
            )
        normalized = _normalized_relative_path(relative_path)
        if normalized in overrides:
            raise ValueError(f"Duplicate tune-group override: {normalized}")
        if not _GROUP_ID_PATTERN.fullmatch(tune_group):
            raise ValueError(
                f"Invalid tune_group {tune_group!r} in {path} line {line_number}"
            )
        overrides[normalized] = TuneGroupOverride(normalized, tune_group)
    return overrides


def automatic_tune_group(relative_path: str) -> str:
    """Conservatively normalize obvious take/version filename variations."""
    stem = Path(_normalized_relative_path(relative_path)).stem
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    text = re.sub(r"\([^)]*\)", " ", ascii_stem.casefold())
    words = re.findall(r"[a-z0-9]+", text)
    normalized_words: list[str] = []
    for word in words:
        word = re.sub(r"(?:gm|xg)$", "", word)
        word = re.sub(r"v?\d+$", "", word)
        if word and word not in _DESCRIPTOR_WORDS and word != "bit":
            normalized_words.append(word)
    group = "".join(normalized_words)
    changed = True
    while changed and group:
        changed = False
        group = re.sub(r"v?\d+$", "", group)
        for suffix in _JOINED_SUFFIXES:
            if group.endswith(suffix) and len(group) > len(suffix):
                group = group[: -len(suffix)]
                changed = True
                break
    if not group:
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
        return f"untitled-{digest}"
    return group


def assign_tune_groups(
    relative_paths: Sequence[str], overrides: dict[str, TuneGroupOverride]
) -> dict[str, str]:
    """Assign every performance to either a manual or automatic tune group."""
    normalized_paths = {_normalized_relative_path(path) for path in relative_paths}
    unknown = sorted(set(overrides) - normalized_paths)
    if unknown:
        raise ValueError(f"Tune-group overrides reference missing files: {unknown}")
    return {
        path: overrides[path].tune_group
        if path in overrides
        else automatic_tune_group(path)
        for path in sorted(normalized_paths, key=str.casefold)
    }


def deterministic_group_splits(
    groups: Sequence[GroupWeight],
    *,
    seed: int = 2026,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, DatasetSplit]:
    """Assign whole tune groups while balancing token, file, and group loads."""
    if len(groups) < len(SPLITS):
        raise ValueError("At least three tune groups are required for three splits")
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be positive and sum to one")
    if len({group.tune_group for group in groups}) != len(groups):
        raise ValueError("Group weights contain duplicate tune_group values")

    total_tokens = sum(group.token_count for group in groups)
    total_files = sum(group.file_count for group in groups)
    target_tokens = dict(
        zip(SPLITS, (max(total_tokens * ratio, 1.0) for ratio in ratios), strict=True)
    )
    target_files = dict(
        zip(SPLITS, (max(total_files * ratio, 1.0) for ratio in ratios), strict=True)
    )
    target_groups = dict(
        zip(SPLITS, (max(len(groups) * ratio, 1.0) for ratio in ratios), strict=True)
    )
    totals = {split: {"tokens": 0, "files": 0, "groups": 0} for split in SPLITS}

    def seeded_order(group: GroupWeight) -> str:
        return hashlib.sha256(f"{seed}:{group.tune_group}".encode()).hexdigest()

    ordered = sorted(
        groups, key=lambda group: (-group.token_count, seeded_order(group))
    )
    assignments: dict[str, DatasetSplit] = {}
    tie_order = list(SPLITS)
    random.Random(seed).shuffle(tie_order)
    tie_rank = {split: index for index, split in enumerate(tie_order)}

    for group in ordered:
        split = min(
            SPLITS,
            key=lambda candidate: (
                0.70 * totals[candidate]["tokens"] / target_tokens[candidate]
                + 0.20 * totals[candidate]["files"] / target_files[candidate]
                + 0.10 * totals[candidate]["groups"] / target_groups[candidate],
                tie_rank[candidate],
            ),
        )
        assignments[group.tune_group] = split
        totals[split]["tokens"] += group.token_count
        totals[split]["files"] += group.file_count
        totals[split]["groups"] += 1
    return assignments


def _token_states(tokens: Sequence[str]) -> list[TokenState]:
    states: list[TokenState] = []
    active: dict[int, deque[int]] = defaultdict(deque)
    pedal_down = False
    velocity_bin: int | None = None

    for token in (*tokens, "__END_SENTINEL__"):
        states.append(
            TokenState(
                active_notes=tuple(
                    (pitch, note_velocity)
                    for pitch in sorted(active)
                    for note_velocity in active[pitch]
                ),
                pedal_down=pedal_down,
                velocity_bin=velocity_bin,
            )
        )
        if token.startswith("VELOCITY_"):
            velocity_bin = int(token.removeprefix("VELOCITY_"))
        elif token.startswith("NOTE_ON_"):
            pitch = int(token.removeprefix("NOTE_ON_"))
            active[pitch].append(velocity_bin if velocity_bin is not None else 0)
        elif token.startswith("NOTE_OFF_"):
            pitch = int(token.removeprefix("NOTE_OFF_"))
            if active[pitch]:
                active[pitch].popleft()
        elif token == PEDAL_DOWN_TOKEN:
            pedal_down = True
        elif token == PEDAL_UP_TOKEN:
            pedal_down = False
    return states


def _state_reconstruction_tokens(
    state: TokenState, *, include_velocity_state: bool
) -> tuple[str, ...]:
    tokens = [BOS_TOKEN]
    if state.pedal_down:
        tokens.append(PEDAL_DOWN_TOKEN)
    by_velocity: dict[int, list[int]] = defaultdict(list)
    for pitch, velocity_bin in state.active_notes:
        by_velocity[velocity_bin].append(pitch)
    last_velocity: int | None = None
    for velocity_bin in sorted(by_velocity):
        tokens.append(f"VELOCITY_{velocity_bin}")
        last_velocity = velocity_bin
        tokens.extend(f"NOTE_ON_{pitch}" for pitch in sorted(by_velocity[velocity_bin]))
    if (
        include_velocity_state
        and state.velocity_bin is not None
        and state.velocity_bin != last_velocity
    ):
        tokens.append(f"VELOCITY_{state.velocity_bin}")
    return tuple(tokens)


def _event_boundaries(tokens: Sequence[str]) -> set[int]:
    boundaries = {
        index
        for index, token in enumerate(tokens)
        if index > 0 and token.startswith("TIME_SHIFT_")
    }
    boundaries.add(len(tokens))
    return boundaries


def _plan_state_safe_chunks(
    tokens: Sequence[str],
    context_length: int,
    states: Sequence[TokenState],
    event_boundaries: set[int],
) -> list[ChunkPlan]:
    ordered_boundaries = sorted(event_boundaries)
    plans: list[ChunkPlan] = []
    start = 0

    while start < len(tokens):
        state = states[start]
        at_event_boundary = start == 0 or start in event_boundaries
        reconstruction = (
            ()
            if start == 0
            else _state_reconstruction_tokens(
                state, include_velocity_state=not at_event_boundary
            )
        )
        capacity = context_length - len(reconstruction)
        if capacity <= 0:
            raise ValueError(
                "State reconstruction is larger than the requested context length"
            )
        maximum_end = min(len(tokens), start + capacity)
        if maximum_end == len(tokens):
            end = len(tokens)
            forced_inside_event_group = False
        else:
            candidates = [
                boundary
                for boundary in ordered_boundaries
                if start < boundary <= maximum_end
            ]
            practical_minimum = start + max(1, capacity // 2)
            safe_candidates = [
                boundary
                for boundary in candidates
                if boundary >= practical_minimum and states[boundary].safe
            ]
            if safe_candidates:
                end = max(safe_candidates)
                forced_inside_event_group = False
            elif candidates:
                end = max(candidates)
                forced_inside_event_group = False
            else:
                end = maximum_end
                forced_inside_event_group = True

        safe_start = start == 0 or (at_event_boundary and state.safe)
        plans.append(
            ChunkPlan(
                start=start,
                end=end,
                context_length=context_length,
                safe_start=safe_start,
                state_reconstruction_required=(
                    start != 0 and (not safe_start or not at_event_boundary)
                ),
                active_notes_at_start=len(state.active_notes),
                pedal_down_at_start=state.pedal_down,
                reconstruction_tokens=reconstruction,
                forced_inside_event_group=forced_inside_event_group,
            )
        )
        start = end
    return plans


def plan_state_safe_chunks(
    tokens: Sequence[str], context_length: int
) -> list[ChunkPlan]:
    """Plan bounded chunks with safe boundaries or explicit state reconstruction."""
    if context_length < 8:
        raise ValueError("context_length must be at least 8 tokens")
    if not tokens:
        return []
    return _plan_state_safe_chunks(
        tokens,
        context_length,
        _token_states(tokens),
        _event_boundaries(tokens),
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_token_array_atomic(path: Path, token_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, token_ids, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _percentage(value: int, total: int) -> float:
    return round(value * 100.0 / total, 6) if total else 0.0


def _record_integer(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Expected integer {key} in generated performance record")
    return value


def _potential_group_reviews(
    group_members: dict[str, list[str]],
    manually_reviewed_paths: set[str],
    limit: int = 50,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    groups = sorted(group_members)
    for index, first in enumerate(groups):
        first_fingerprint = first.replace("-", "")
        if len(first_fingerprint) < 6:
            continue
        for second in groups[index + 1 :]:
            second_fingerprint = second.replace("-", "")
            if len(second_fingerprint) < 6:
                continue
            similarity = SequenceMatcher(
                None, first_fingerprint, second_fingerprint
            ).ratio()
            if similarity < 0.88:
                continue
            if set(group_members[first] + group_members[second]).issubset(
                manually_reviewed_paths
            ):
                continue
            candidates.append(
                {
                    "first_group": first,
                    "first_files": group_members[first],
                    "second_group": second,
                    "second_files": group_members[second],
                    "similarity": round(similarity, 6),
                }
            )
    return sorted(
        candidates,
        key=lambda item: (
            item["similarity"] if isinstance(item["similarity"], float) else 0.0
        ),
        reverse=True,
    )[:limit]


def _markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Grouped tokenized dataset preparation",
        "",
        f"- Files: **{report['file_count']}**",
        f"- Tune groups: **{report['tune_group_count']}**",
        f"- Total tokens: **{report['total_token_count']}**",
        f"- Leakage-free: **{report['group_leakage_free']}**",
        f"- Token arrays verified: **{report['token_array_round_trip_verified']}**",
        "",
        "## Split balance",
        "",
        "| Split | Files | Groups | Tokens | Token share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    split_balance = report["split_balance"]
    assert isinstance(split_balance, dict)
    for split in SPLITS:
        item = split_balance[split]
        assert isinstance(item, dict)
        lines.append(
            f"| {split} | {item['file_count']} | {item['tune_group_count']} | "
            f"{item['token_count']} | {item['token_percent']}% |"
        )

    context_analysis = report["context_analysis"]
    assert isinstance(context_analysis, dict)
    lines.extend(
        [
            "",
            "## Chunk analysis",
            "",
            "| Context | Chunks | Safe starts | State carry | Pedal carry | "
            "Forced event splits |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for context, item in context_analysis.items():
        assert isinstance(item, dict)
        lines.append(
            f"| {context} | {item['chunk_count']} | "
            f"{item['safe_noninitial_chunk_percent']}% | "
            f"{item['state_reconstruction_chunk_count']} | "
            f"{item['pedal_carry_chunk_count']} | "
            f"{item['forced_inside_event_group_count']} |"
        )

    reviews = report["potential_grouping_reviews"]
    assert isinstance(reviews, list)
    lines.extend(["", "## Grouping review candidates", ""])
    if reviews:
        for item in reviews:
            assert isinstance(item, dict)
            lines.append(
                f"- `{item['first_group']}` vs `{item['second_group']}` "
                f"(similarity {item['similarity']})"
            )
    else:
        lines.append("No high-similarity ungrouped names were found.")
    lines.append("")
    return "\n".join(lines)


def prepare_tokenized_dataset(
    *,
    input_root: Path,
    output_root: Path,
    group_overrides_path: Path,
    dataset_manifest_path: Path,
    report_json_path: Path,
    report_markdown_path: Path,
    config: TokenizerConfig,
    seed: int = 2026,
    context_lengths: tuple[int, ...] = (2048,),
    overwrite: bool = False,
) -> dict[str, object]:
    """Tokenize complete performances, split tune groups, and analyze chunks."""
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Cleaned MIDI directory not found: {input_root}")
    if not context_lengths or any(length < 8 for length in context_lengths):
        raise ValueError("Context lengths must all be at least 8")
    files = sorted(
        (
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mid"
        ),
        key=lambda path: path.relative_to(input_root).as_posix().casefold(),
    )
    relative_paths = [path.relative_to(input_root).as_posix() for path in files]
    overrides = load_tune_group_overrides(group_overrides_path)
    tune_groups = assign_tune_groups(relative_paths, overrides)
    tokenizer = ExpressiveMidiTokenizer(config)
    performance_data: list[dict[str, object]] = []

    for path, relative_path in zip(files, relative_paths, strict=True):
        encoded = tokenizer.encode_file(path)
        token_ids = tokenizer.tokens_to_ids(encoded.tokens)
        output_path = output_root / Path(relative_path).with_suffix(".npy")
        if output_path.exists() and not overwrite:
            existing = np.load(output_path, allow_pickle=False)
            if not np.array_equal(existing, token_ids):
                raise FileExistsError(
                    f"Token array differs; rerun with --overwrite: {output_path}"
                )
        else:
            _write_token_array_atomic(output_path, token_ids)
        reloaded = np.load(output_path, allow_pickle=False)
        if tokenizer.ids_to_tokens(reloaded) != encoded.tokens:
            raise ValueError(f"Token-array round trip failed: {relative_path}")
        performance_data.append(
            {
                "source_file": relative_path,
                "source_path": _display_path(path),
                "tune_group": tune_groups[relative_path],
                "token_count": len(encoded.tokens),
                "note_count": encoded.diagnostics.event_counts.get("note_on", 0),
                "token_array_path": _display_path(output_path),
            }
        )

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in performance_data:
        by_group[str(item["tune_group"])].append(item)
    group_weights = [
        GroupWeight(
            tune_group=group,
            file_count=len(items),
            token_count=sum(_record_integer(item, "token_count") for item in items),
        )
        for group, items in by_group.items()
    ]
    split_by_group = deterministic_group_splits(group_weights, seed=seed)
    manifest_rows: list[dict[str, object]] = []
    context_totals = {
        length: Counter(
            {
                "chunk_count": 0,
                "noninitial_chunk_count": 0,
                "safe_noninitial_chunk_count": 0,
                "state_reconstruction_chunk_count": 0,
                "active_note_carry_chunk_count": 0,
                "pedal_carry_chunk_count": 0,
                "reconstruction_token_count": 0,
                "forced_inside_event_group_count": 0,
            }
        )
        for length in context_lengths
    }
    context_by_split = {
        length: {split: Counter() for split in SPLITS} for length in context_lengths
    }
    files_ending_pedal_down: list[str] = []

    for item in performance_data:
        split = split_by_group[str(item["tune_group"])]
        token_ids = np.load(str(item["token_array_path"]), allow_pickle=False)
        tokens = tokenizer.ids_to_tokens(token_ids)
        states = _token_states(tokens)
        event_boundaries = _event_boundaries(tokens)
        if states[-1].pedal_down:
            files_ending_pedal_down.append(str(item["source_file"]))
        chunk_summary: dict[str, dict[str, int]] = {}
        for context_length in context_lengths:
            plans = _plan_state_safe_chunks(
                tokens, context_length, states, event_boundaries
            )
            counts = Counter(
                {
                    "chunk_count": len(plans),
                    "noninitial_chunk_count": max(0, len(plans) - 1),
                    "safe_noninitial_chunk_count": sum(
                        plan.safe_start for plan in plans[1:]
                    ),
                    "state_reconstruction_chunk_count": sum(
                        plan.state_reconstruction_required for plan in plans
                    ),
                    "active_note_carry_chunk_count": sum(
                        plan.active_notes_at_start > 0 for plan in plans
                    ),
                    "pedal_carry_chunk_count": sum(
                        plan.pedal_down_at_start for plan in plans
                    ),
                    "reconstruction_token_count": sum(
                        len(plan.reconstruction_tokens) for plan in plans
                    ),
                    "forced_inside_event_group_count": sum(
                        plan.forced_inside_event_group for plan in plans
                    ),
                }
            )
            context_totals[context_length].update(counts)
            context_by_split[context_length][split].update(counts)
            chunk_summary[str(context_length)] = dict(counts)
        manifest_rows.append({**item, "split": split, "chunks": chunk_summary})

    total_tokens = sum(_record_integer(item, "token_count") for item in manifest_rows)
    split_balance: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        split_rows = [item for item in manifest_rows if item["split"] == split]
        split_groups = {str(item["tune_group"]) for item in split_rows}
        split_tokens = sum(_record_integer(item, "token_count") for item in split_rows)
        split_balance[split] = {
            "file_count": len(split_rows),
            "file_percent": _percentage(len(split_rows), len(manifest_rows)),
            "tune_group_count": len(split_groups),
            "tune_group_percent": _percentage(len(split_groups), len(by_group)),
            "token_count": split_tokens,
            "token_percent": _percentage(split_tokens, total_tokens),
            "chunk_counts": {
                str(length): context_by_split[length][split]["chunk_count"]
                for length in context_lengths
            },
        }

    context_report: dict[str, dict[str, object]] = {}
    for length in context_lengths:
        counts = context_totals[length]
        noninitial = counts["noninitial_chunk_count"]
        context_report[str(length)] = {
            **dict(counts),
            "safe_noninitial_chunk_percent": _percentage(
                counts["safe_noninitial_chunk_count"], noninitial
            ),
            "state_reconstruction_chunk_percent": _percentage(
                counts["state_reconstruction_chunk_count"], noninitial
            ),
            "mean_chunks_per_file": round(
                counts["chunk_count"] / len(manifest_rows), 6
            ),
            "split_chunk_counts": {
                split: context_by_split[length][split]["chunk_count"]
                for split in SPLITS
            },
        }

    group_splits = {
        group: {
            str(item["split"]) for item in manifest_rows if item["tune_group"] == group
        }
        for group in by_group
    }
    leakage = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    report: dict[str, object] = {
        "settings": {
            "tokenizer": config.to_dict(),
            "split_seed": seed,
            "target_split_ratios": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1,
            },
            "context_lengths": list(context_lengths),
        },
        "file_count": len(manifest_rows),
        "tune_group_count": len(by_group),
        "manual_override_file_count": len(overrides),
        "multi_performance_group_count": sum(
            len(items) > 1 for items in by_group.values()
        ),
        "total_token_count": total_tokens,
        "total_note_count": sum(
            _record_integer(item, "note_count") for item in manifest_rows
        ),
        "token_dtype": "uint16",
        "full_performances_stored_separately": True,
        "token_array_round_trip_verified": True,
        "group_leakage_free": not leakage,
        "group_leakage": leakage,
        "split_balance": split_balance,
        "context_analysis": context_report,
        "state_safe_chunk_plans_verified": all(
            item["forced_inside_event_group_count"] == 0
            for item in context_report.values()
        ),
        "files_ending_with_sustain_down_count": len(files_ending_pedal_down),
        "files_ending_with_sustain_down": files_ending_pedal_down,
        "ending_sustain_sequences_modified": False,
        "potential_grouping_reviews": _potential_group_reviews(
            {
                group: sorted(str(item["source_file"]) for item in items)
                for group, items in by_group.items()
            },
            set(overrides),
        ),
        "token_array_file_count": len(manifest_rows),
    }
    manifest_content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(
            manifest_rows, key=lambda item: str(item["source_file"]).casefold()
        )
    )
    _write_text_atomic(dataset_manifest_path, manifest_content)
    _write_text_atomic(
        report_json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(report_markdown_path, _markdown_report(report))
    return report
