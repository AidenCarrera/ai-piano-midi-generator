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
