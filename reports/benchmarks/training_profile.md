# PyTorch training profile

## Method

- PyTorch 2.13.0+cu130 on an NVIDIA GeForce RTX 2060 SUPER
- Eager training with CUDA AMP and fused AdamW
- Physical batch 8, context length 2,048, gradient accumulation 1
- Five unprofiled warmup steps followed by ten profiled optimizer steps
- Fixed-shape batches and the in-memory token-array cache enabled
- Operator shapes, CUDA activity, FLOPs, and tensor memory recorded

Profiler instrumentation adds overhead. Percentages below use device activity
inside the `training_step` range rather than instrumented wall time.

## Results

The measured training step averaged 203.63 ms of device activity and peaked at
1,664.79 MiB of allocated GPU memory.

| Component | Device time per step | Training-step share |
|---|---:|---:|
| Efficient-attention forward and backward | 89.88 ms | 44.14% |
| Matrix multiplication (`mm` and `addmm`) | 66.28 ms | 32.55% |
| Layer normalization | 10.79 ms | 5.30% |
| GELU forward and backward | 6.78 ms | 3.33% |
| Optimizer step | 1.32 ms | 0.65% |
| Gradient clipping | 1.11 ms | 0.55% |
| Host-to-device transfer | 0.04 ms | 0.02% |

Cached data loading used 0.76 ms of CPU time per step. It is not a material
bottleneck relative to the 203.63 ms of device work.

## Decisions

- Keep exact GELU. Even eliminating all GELU cost would cap the improvement at
  3.33%; approximation would recover only part of that.
- Keep gradient clipping. Its 0.55% cost is too small to justify removing a
  training-stability safeguard.
- Keep fused AdamW. Optimizer work is already only 0.65% of device time.
- Do not tune data loading further. Cached loading and transfer are negligible.
- Prioritize making `torch.compile`/Triton available and benchmark compiler
  modes. Attention and matrix multiplication account for 76.69% of device time.
- Retain PyTorch SDPA dispatch. It selected the memory-efficient CUTLASS SM75
  attention kernels appropriate to the current Turing GPU.

## Reproduction

```powershell
uv run python scripts/experiments/profile_training.py --device cuda --warmup-steps 5 --profile-steps 10
```

The machine-readable results are in `training_profile.json`. The detailed
Chrome trace is generated locally at `artifacts/profiler/training_trace.json`.
