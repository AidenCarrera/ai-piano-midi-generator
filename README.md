# AI Piano MIDI Generator

Environment and project configuration for an AI jazz piano MIDI generator.
This setup uses Python 3.13, `uv`, and a CUDA-enabled PyTorch build for Windows.

## Project layout

```text
configs/                    Versioned training, generation, and experiment settings
data/raw/                   Read-only source MIDI files
data/processed/             Cleaned MIDI and compact token arrays
data/manifests/             Review decisions, inventories, groups, and splits
notebooks/                  Optional interactive report exploration
scripts/                    Stable dataset, training, generation, and evaluation CLIs
scripts/experiments/        Reproducible model-analysis experiments
src/ai_piano_midi_generator/ Reusable library code
tests/                      Automated tests
checkpoints/                Local model checkpoints, organized by model
outputs/                    Local generated MIDI files, organized by model
artifacts/token-arrays/     Generated token arrays, mirroring outputs/ runs
reports/                    Generated reports by pipeline phase and model
reports/figures/            Generated dataset and experiment figures
```

Generated token arrays live under `artifacts/token-arrays/` with the same model
and experiment subdirectories as their MIDI files. Checkpoints, outputs,
artifacts, generated reports, raw data, and processed data are intentionally
ignored by Git. The active model directories are `baseline_2048` and
`eos_2048`.

## Setup

From the project root, create or update the local environment and install all
runtime and development dependencies:

```powershell
uv sync --locked
```

The committed `uv.lock` is the reproducibility boundary for dependency
versions. Use plain `uv sync` only when intentionally updating the lockfile.

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
uv lock --check
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
- `reports/dataset/inventory.json`
- `reports/dataset/inventory.md`

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
`reports/dataset/validation.json` and `reports/dataset/validation.md`,
reparsing every expected output and
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
binary CC64 pedal tokens. CC64 values that do not change the binary pedal state
emit no token, matching the sampler's grammar. Tempo changes are converted to
elapsed seconds rather than represented as tokens. Decoding writes a fixed 120 QPM representation for
round-trip validation only.

Use `--timing-resolution`, `--maximum-time-shift`, `--velocity-bins`, and
`--decode-tempo` to evaluate other settings. The command writes
`reports/tokenization/analysis.json` and `reports/tokenization/analysis.md`.

## Prepare grouped token arrays and splits

Create deterministic, leakage-resistant train, validation, and test assignments
and one compact NumPy token array per complete performance:

```powershell
uv run python scripts/prepare_tokenized_dataset.py
```

The command groups related takes before assigning an 80/10/10 split with seed
`2026`. Manual filename-family decisions live in the versionable
`data/manifests/tune_group_overrides.jsonl` file:

```json
{"relative_path":"alternate-name.mid","tune_group":"canonical-tune-name"}
```

Generated arrays are written under `data/processed/tokens`, with metadata in
`data/manifests/tokenized_dataset.jsonl`. Complete performances remain separate;
the preparation report analyzes the active 2,048-token chunk plan by default.
Unsafe chunk starts receive explicit active-note and pedal-state reconstruction
prefixes. Existing arrays are verified and reused when identical; pass
`--overwrite` to replace differing generated arrays. Arrays written before
redundant pedal events were dropped differ from current encoder output; rerun
with `--overwrite` and retrain to train on the canonical representation.
The preparation reports are written under `reports/tokenization/`.

## Train the decoder-only baseline

Run the required checks before a full 2,048-context training job:

```powershell
uv run python scripts/train.py --smoke-test
uv run python scripts/train.py --tiny-overfit
```

Start or resume the full plain-PyTorch baseline with:

```powershell
uv run python scripts/train.py
uv run python scripts/train.py --resume checkpoints/baseline_2048/latest.pt
```

The active baseline uses a 2,048-token causal context, shifted next-token loss,
AdamW, warmup plus cosine decay, gradient accumulation and clipping, CUDA AMP
and fused AdamW when enabled and available, seeded random-number generators,
token-weighted validation over the whole validation split (so losses do not
depend on batch size), and latest/best checkpoints. Logged training and
validation metrics are appended to `training_history.jsonl` beside the training
report for learning-curve figures. Strict deterministic
kernels are available through `training.deterministic` but are disabled in the
performance-oriented baseline configuration. Set `training.compile_model` to
`true` to compile the model in place with `dynamic=False`; `training.compile_mode`
accepts `default`, `reduce-overhead`, `max-autotune`, or
`max-autotune-no-cudagraphs`. Compilation is opt-in because distinct padded
sequence lengths can trigger recompilation. Training and validation sequences
are therefore padded to the configured context length; the training loader also
drops its incomplete final batch so compiled training steps retain one input
shape. Validation keeps its incomplete final batch so every validation chunk is
measured. The baseline leaves compilation disabled so training also works
without a supported Triton installation. Training reports record the resolved
model/training configuration, dependency versions, and a SHA-256 digest of the
dataset manifest. Training loss and token counters remain on the training
device between log intervals; CUDA is synchronized only at explicit timing and
metric boundaries so throughput measurements include completed GPU work. A
resumed report leaves fresh-model loss reduction unset because it would not be
a meaningful comparison.

Token arrays are loaded once into a read-only per-split memory cache instead of
reopening an `.npy` file for every chunk. On the current Windows/CUDA system,
the fixed-batch loader benchmark selected zero workers: its cached median epoch
was 0.0267 seconds, versus 0.0544 with two workers and 0.0553 with four. The
repeatable benchmark is `scripts/experiments/benchmark_data_loader.py`, with
results stored in `reports/benchmarks/data_loader_cache.json`.

The representative PyTorch profiler capture is reproducible with
`scripts/experiments/profile_training.py`. Its summary is stored in
`reports/benchmarks/training_profile.md`: attention and matrix multiplication
accounted for 76.69% of device time, while GELU, gradient clipping, the fused
optimizer, cached loading, and host-to-device transfer were each too small to
justify isolated micro-optimizations.

The explicit equivalents are:

```powershell
uv run python scripts/train.py --config configs/training_baseline_2048.toml --smoke-test
uv run python scripts/train.py --config configs/training_baseline_2048.toml --tiny-overfit
uv run python scripts/train.py --config configs/training_baseline_2048.toml
```

Its checkpoints are written under `checkpoints/baseline_2048`.

Checkpoints include optimizer and random-number-generator state and should only
be loaded from trusted sources: full training-state checkpoints rely on Python
pickle through PyTorch. Resume restores those states, but it does not restore a
mid-epoch shuffled-data cursor. Runs are therefore repeatable from the start on
the same software/hardware stack, while a mid-run resume is trajectory-close
rather than bitwise identical. PyTorch does not promise identical results
across releases, devices, or platforms.

Train the same architecture and 6,000-step schedule from scratch with
EOS-ending excerpts mixed into 20% of training draws:

```powershell
uv run python scripts/train.py --config configs/training_eos_2048.toml --smoke-test
uv run python scripts/train.py --config configs/training_eos_2048.toml
```

This run does not load baseline weights or alter the tokenizer, vocabulary,
token IDs, validation split, or checkpoint format. Its outputs are isolated
under `checkpoints/eos_2048/` and `reports/eos_2048/`.

Tail construction never inserts synthetic note-off or pedal-up events before
the training EOS. If the model emits EOS with notes or sustain still active,
the inference exporter deterministically closes that state before decoding and
records the cleanup in the generation report.

Generate a constrained MIDI sample from a checkpoint:

```powershell
uv run python scripts/generate.py --checkpoint checkpoints/baseline_2048/best.pt
```

Sampling supports temperature, top-k, and top-p controls and blocks invalid
note-off and pedal transitions, including note-offs that would close a note in
the step it started. EOS remains eligible after the configured minimum; any
active note or sustain state is closed by the exporter before decoding through
the existing tokenizer. Once a prompt plus generation exceeds the context, the
model window is laid out like a training chunk: BOS, explicit held-note and
pedal state, then the most recent tokens from an event boundary.

Diagnose long-context time-shift behavior against length-matched train,
validation, and test windows using the capped samples in the baseline natural-
termination report:

```powershell
uv run python scripts/experiments/diagnose_time_shift_collapse.py
```

`endurance` always samples exactly 2,048 model tokens and rejects duration or
silence safeguards. `musical` permits safe EOS after its configured minimum and
can terminate at explicit duration or note-on-silence limits. Safeguards are
disabled by default in the sampler and general generation CLI; each
intervention is recorded. To opt into the train-derived musical limits for one
sample:

```powershell
uv run python scripts/generate.py --checkpoint checkpoints/baseline_2048/best.pt --config configs/generation_musical_2048.toml
```

Use `--mode endurance --max-new-tokens 2048` for an unconstrained stress test,
or configure musical controls directly with `--max-duration-seconds`,
`--max-silence-seconds`, and `--safeguard-action`. Constrained musical results
do not replace the endurance evidence or demonstrate that the underlying model
attractor is fixed.

Evaluate natural termination without any minimum length or timing safeguard:

```powershell
uv run python scripts/experiments/evaluate_natural_termination.py
```

This runs 10 seeds at temperatures 0.85, 0.90, and 0.95 for both unconditional
generation and continuations from the first 256 tokens of distinct held-out
test files. EOS
is legal from the first prediction; 2,048 tokens is only a maximum. The report
records natural EOS position, cap endings, notes, duration, state repairs, and
dataset-derived degeneracy flags. If interrupted, continue with `--resume`.
The generated DAW audition queue deliberately keeps automated validity and
human judgments such as completeness, repetition, and harmonic coherence
separate. MIDI files are written under `outputs/`; NumPy token traces used by
the diagnostics are written under `artifacts/token-arrays/`.

Evaluate the best checkpoint once on the held-out test split and generate
matched unconditional and prompted samples with:

```powershell
uv run python scripts/evaluate.py
```

The evaluation defaults to `checkpoints/baseline_2048/best.pt`, writes JSON and
Markdown reports under `reports/baseline_2048/`, and writes MIDI samples under
`outputs/baseline_2048/evaluation/`. EOS is allowed immediately by default;
use `--min-new-tokens` only when an explicitly forced minimum is desired.

## Visualize datasets and experiments

Create report-driven PNG figures without changing the dataset, tokenizer,
model, or training pipeline:

```powershell
uv run python scripts/create_visualizations.py
```

The command saves dataset preparation, tokenization, grouped split, learning
curve and test loss, generated-versus-test, and musical-statistics (pitch class,
register, notes per onset) figures under `reports/figures/`. Runs trained before
`training_history.jsonl` existed are shown as endpoints only.
Missing optional experiment reports are skipped unless `--strict` is supplied.
Use `--format svg` or `--format both` for vector output.

For interactive exploration, start Jupyter Lab and open
`notebooks/project_visualizations.ipynb`:

```powershell
uv run --with jupyter jupyter lab
```
