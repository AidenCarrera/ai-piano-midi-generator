"""Generate a constrained token sequence and decode it to valid MIDI."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
import torch
from symusic.core import ScoreTick

from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.sampling import sample_tokens
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import load_checkpoint, select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--token-output", type=Path)
    parser.add_argument("--prompt-array", type=Path)
    parser.add_argument("--prompt-midi", type=Path)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--min-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--mode", choices=["endurance", "musical"])
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument("--max-silence-seconds", type=float)
    parser.add_argument("--safeguard-action", choices=["eos", "mask_time_shifts"])
    return parser


def _generation_settings(args: argparse.Namespace) -> dict[str, object]:
    configured: dict[str, object] = {}
    if args.config is not None:
        raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
        value = raw.get("generation", {})
        if not isinstance(value, dict):
            raise ValueError("Config [generation] must be a table")
        configured = value

    def setting(argument: str, config_key: str, default: object) -> object:
        value = getattr(args, argument)
        return value if value is not None else configured.get(config_key, default)

    return {
        "generation_mode": setting("mode", "mode", "musical"),
        "maximum_new_tokens": setting("max_new_tokens", "maximum_new_tokens", 256),
        "minimum_new_tokens": setting("min_new_tokens", "minimum_new_tokens", 0),
        "temperature": setting("temperature", "temperature", 0.95),
        "top_k": setting("top_k", "top_k", 40),
        "top_p": setting("top_p", "top_p", 0.95),
        "maximum_duration_seconds": setting(
            "max_duration_seconds", "maximum_duration_seconds", None
        ),
        "maximum_silence_seconds": setting(
            "max_silence_seconds", "maximum_silence_seconds", None
        ),
        "safeguard_action": setting("safeguard_action", "safeguard_action", "eos"),
    }


def _integer_setting(settings: dict[str, object], name: str) -> int:
    value = settings[name]
    if not isinstance(value, int):
        raise ValueError(f"Generation setting {name} must be an integer")
    return value


def _float_setting(settings: dict[str, object], name: str) -> float:
    value = settings[name]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Generation setting {name} must be numeric")
    return float(value)


def _optional_float_setting(settings: dict[str, object], name: str) -> float | None:
    value = settings[name]
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Generation setting {name} must be numeric or null")
    return float(value)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _generation_settings(args)
    run_name = args.checkpoint.parent.name
    output_path = args.output or Path("outputs") / run_name / "generated.mid"
    report_path = args.report or Path("reports") / run_name / "generation.json"
    mode_value = settings["generation_mode"]
    action_value = settings["safeguard_action"]
    if mode_value not in ("endurance", "musical"):
        raise ValueError("Generation mode must be endurance or musical")
    if action_value not in ("eos", "mask_time_shifts"):
        raise ValueError("Unknown safeguard action")
    mode = cast(Literal["endurance", "musical"], mode_value)
    action = cast(Literal["eos", "mask_time_shifts"], action_value)
    if args.prompt_array and args.prompt_midi:
        raise ValueError("Choose only one prompt source")
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_settings = checkpoint.get("model_config")
    if not isinstance(model_settings, dict):
        raise ValueError("Checkpoint is missing model_config")
    model = PianoTransformer(TransformerConfig(**model_settings)).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()
    tokenizer = ExpressiveMidiTokenizer()

    if args.prompt_array:
        prompt_ids = np.load(args.prompt_array, allow_pickle=False)
        prompt_tokens = tokenizer.ids_to_tokens(prompt_ids)[: args.prompt_length]
    elif args.prompt_midi:
        prompt_tokens = tokenizer.encode_file(args.prompt_midi).tokens[
            : args.prompt_length
        ]
    else:
        prompt_tokens = (BOS_TOKEN,)
    if not prompt_tokens or prompt_tokens[0] != BOS_TOKEN:
        prompt_tokens = (BOS_TOKEN, *prompt_tokens)

    sampled = sample_tokens(
        model,
        tokenizer,
        prompt_tokens=prompt_tokens,
        maximum_new_tokens=_integer_setting(settings, "maximum_new_tokens"),
        minimum_new_tokens=_integer_setting(settings, "minimum_new_tokens"),
        temperature=_float_setting(settings, "temperature"),
        top_k=_integer_setting(settings, "top_k"),
        top_p=_float_setting(settings, "top_p"),
        seed=args.seed,
        constrain_states=True,
        generation_mode=mode,
        maximum_duration_seconds=_optional_float_setting(
            settings, "maximum_duration_seconds"
        ),
        maximum_silence_seconds=_optional_float_setting(
            settings, "maximum_silence_seconds"
        ),
        safeguard_action=action,
    )
    decoded = tokenizer.decode(sampled.tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded.score.dump_midi(output_path)
    if args.token_output is not None:
        args.token_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            args.token_output,
            tokenizer.tokens_to_ids(sampled.tokens),
            allow_pickle=False,
        )
    reparsed = ScoreTick.from_file(output_path)
    report: dict[str, object] = {
        "checkpoint": args.checkpoint.as_posix(),
        "output": output_path.as_posix(),
        "token_output": (
            args.token_output.as_posix() if args.token_output is not None else None
        ),
        "prompt_token_count": len(prompt_tokens),
        "generated_total_token_count": len(sampled.tokens),
        "generation_settings": settings,
        "note_count": sum(len(track.notes) for track in reparsed.tracks),
        "cc64_event_count": sum(
            1
            for track in reparsed.tracks
            for control in track.controls
            if control.number == 64
        ),
        "invalid_events_blocked": sampled.invalid_events_blocked,
        "final_state_repairs": sampled.final_state_repairs,
        "safeguard_interventions": list(sampled.safeguard_interventions),
        "decoder_repairs": decoded.repairs.to_dict(),
        "midi_parse_valid": True,
        "device": str(device),
    }
    _write_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if decoded.repairs.total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
