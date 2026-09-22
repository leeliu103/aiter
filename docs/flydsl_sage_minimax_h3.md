# FlyDSL SageAttention for MiniMax-H3

The [gfx1201 kernel](../aiter/ops/flydsl/kernels/sage_attention_gfx1201.py)
uses head-major ordering: consecutive block IDs cover query tiles of the
same head to improve K/V cache reuse. The builder API, attention arithmetic,
and grid size are unchanged.

## Benchmark

Requires a gfx1201 GPU, ROCm PyTorch, and FlyDSL. Tested with Python 3.12,
PyTorch `2.11.0+gitd0c8b1f`, and FlyDSL `0.2.0`. The script loads the kernel
files directly; no AITER rebuild or model weights are needed.

From the repository root:

```bash
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python3 op_tests/op_benchmarks/flydsl/bench_sage_minimax_h3.py
```

Runs SP2/SP4/SP8. Add `--sp 2` to select SP2 or `--repeats 3` for fewer calls.
The script compares the [original commit](https://github.com/leeliu103/aiter/commit/c3184c21b5619507a444f81c775ad78c09bb0566)
with the optimized kernel in the checkout. It reads the baseline with
`git show`; shallow clones must include it (`git fetch --unshallow`).

Each workload represents one H3 rank after sequence-parallel redistribution:
`[B, S, H, D] = [1, 114660, 56 / SP, 128]`, padded to 114688 tokens.
Both kernels use 128x32 tiles, LDS padding 8, K/V prefetch,
`gated_o_rescale=True`, a 32 KiB LDS reservation, and key-tail peeling.

Seed-42 synthetic BF16 inputs are prepared once, with K smoothing and shared
INT8 Q/K and FP8 V buffers; BF16 outputs use separate buffers. By default,
each kernel gets two warmups and five timed calls, with alternating execution
order. The script reports median GPU-event times and checks every timed
result's valid tokens for finite, byte-identical values against the original.
Compilation, preparation, and output checks are outside timing.

## Results

Measured on one gfx1201 GPU, 2026-09-22:

| Workload | Heads per rank | Original (ms) | Head-major (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| SP2 | 28 | 1794.530 | 1019.604 | 1.760x |
| SP4 | 14 | 892.497 | 513.598 | 1.738x |
| SP8 | 7 | 394.115 | 257.942 | 1.528x |

All checks passed. These core timings exclude communication and model
inference. Performance on other shapes and model output quality were not evaluated.

Baseline source: `amd-aiter==0.1.16.post3+gfx1201.g693c3b6bc8`; its origin and
verified SHA256 are recorded in the baseline commit message. AMD copyright
and MIT notices are retained; see [LICENSE](../LICENSE).
