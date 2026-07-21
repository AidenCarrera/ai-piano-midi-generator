# AI Piano MIDI Generator

Environment and project configuration for an AI jazz piano MIDI generator.
This setup uses Python 3.13, `uv`, and a CUDA-enabled PyTorch build for Windows.

## Setup

From the project root, create or update the local environment and install all
runtime and development dependencies:

```powershell
uv sync
```

## Verify CUDA

Confirm that PyTorch is using its CUDA build and can access the NVIDIA GPU:

```powershell
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Quality checks

Format and lint the project with Ruff:

```powershell
uv run ruff format .
uv run ruff check .
```

Run static type checking with Pyright:

```powershell
uv run pyright
```

Run the test suite with pytest:

```powershell
uv run pytest
```

## Jupyter Lab

Start Jupyter Lab without adding it to the project dependencies:

```powershell
uv run --with jupyter jupyter lab
```

## Inspect the raw MIDI dataset

Run the read-only dataset inspector from the project root:

```powershell
uv run python scripts/inspect_dataset.py --input data/raw/bushgraft
```

If the MIDI files are stored directly in `data/raw`, use `--input data/raw`
instead.

The inspector recursively reads `.mid` files without changing them and writes:

- `data/manifests/raw_inventory.jsonl`
- `reports/dataset_summary.json`
- `reports/dataset_summary.md`

Files are accepted when they parse and contain valid pitched notes on General
MIDI piano programs 0-7. Files with usable piano notes plus drums, non-piano
programs, or invalid notes need review. Unparseable files and files without
valid pitched notes are rejected.

## Prepare piano-only MIDI files

Preview conservative piano-track selection without writing processed files:

```powershell
uv run python scripts/prepare_dataset.py --dry-run
```

After reviewing ambiguous files, write cleaned MIDI copies and the processed
inventory, then validate every final output:

```powershell
uv run python scripts/prepare_dataset.py
```

Existing processed MIDI files are skipped by default. Pass `--overwrite` only
when they should be replaced. Raw files are always read-only.

Automatic selection requires exactly one non-drum track using a General MIDI
piano program (0-7). Add one JSON object per line to
`data/manifests/track_selections.jsonl` to resolve ambiguous files:

```json
{"relative_path":"example.mid","selected_track_indices":[1]}
{"relative_path":"two-pianos.mid","selected_track_indices":[0,2],"merge":true}
{"relative_path":"pedal-carrier.mid","selected_track_indices":[1,2],"cc64_track_indices":[0],"merge":true}
{"relative_path":"exclude.mid","skip":true}
{"relative_path":"unusable.mid","reject":true,"reason":"No identifiable piano track."}
```

Selecting multiple tracks requires `"merge": true`; they are never merged
automatically. Preparation preserves tick timing, durations, velocities, tempo
events, and CC64 sustain events. It only removes non-positive-duration notes
from explicitly or automatically selected piano tracks.

Use `cc64_track_indices` only when manual review identifies a discarded
controller-carrier track whose sustain events belong to the selected piano.
Explicitly rejected files require a reason. A completed non-dry run writes
`reports/final_dataset_validation.json` and
`reports/final_dataset_validation.md`, reparsing every expected output and
checking statistics, filename traceability, missing files, and invalid notes.
Valid MIDI pitches outside the standard 88-key acoustic-piano range (21-108)
are retained but listed as non-blocking suspicious outputs for human review.

## Analyze expressive tokenization

Analyze the cleaned MIDI files in memory without writing a tokenized training
dataset:

```powershell
uv run python scripts/analyze_tokenization.py
```

The default event vocabulary uses 10 ms time steps, time-shift tokens up to one
second, 32 velocity bins, note-on and note-off tokens for all MIDI pitches, and
binary CC64 pedal tokens. Tempo changes are converted to elapsed seconds rather
than represented as tokens. Decoding writes a fixed 120 QPM representation for
round-trip validation only.

Use `--timing-resolution`, `--maximum-time-shift`, `--velocity-bins`, and
`--decode-tempo` to evaluate other settings. The command writes
`reports/tokenization_analysis.json` and `reports/tokenization_analysis.md`.
