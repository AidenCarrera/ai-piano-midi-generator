"""Generate a constrained token sequence and decode it to valid MIDI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from symusic.core import ScoreTick

from ai_piano_midi_generator.model import PianoTransformer, TransformerConfig
from ai_piano_midi_generator.sampling import sample_tokens
from ai_piano_midi_generator.tokenizer import BOS_TOKEN, ExpressiveMidiTokenizer
from ai_piano_midi_generator.training import load_checkpoint, select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/generated.mid"))
    parser.add_argument("--report", type=Path, default=Path("reports/generation.json"))
    parser.add_argument("--prompt-array", type=Path)
    parser.add_argument("--prompt-midi", type=Path)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        maximum_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        constrain_states=True,
    )
    decoded = tokenizer.decode(sampled.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    decoded.score.dump_midi(args.output)
    reparsed = ScoreTick.from_file(args.output)
    report: dict[str, object] = {
        "checkpoint": args.checkpoint.as_posix(),
        "output": args.output.as_posix(),
        "prompt_token_count": len(prompt_tokens),
        "generated_total_token_count": len(sampled.tokens),
        "note_count": sum(len(track.notes) for track in reparsed.tracks),
        "cc64_event_count": sum(
            1
            for track in reparsed.tracks
            for control in track.controls
            if control.number == 64
        ),
        "invalid_events_blocked": sampled.invalid_events_blocked,
        "final_state_repairs": sampled.final_state_repairs,
        "decoder_repairs": decoded.repairs.to_dict(),
        "midi_parse_valid": True,
        "device": str(device),
    }
    _write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if decoded.repairs.total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
