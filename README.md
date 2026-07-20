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
