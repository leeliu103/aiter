#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare the unchanged Sage core with head-major scheduling on MiniMax-H3."""

import argparse
import importlib.util
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import flydsl
import torch

ROOT = Path(__file__).resolve().parents[3]
KERNEL = Path("aiter/ops/flydsl/kernels/sage_attention_gfx1201.py")
BASELINE = "c3184c21b5619507a444f81c775ad78c09bb0566"
SEQ_LEN = 114660


def load_builder(path, name):
    # Load these exact files, without importing an installed AITER kernel.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_sage_attention_v2_core


def make_inputs(heads, seq_len=SEQ_LEN):
    """Prepare synthetic BF16 Q/K/V once; all preprocessing is outside timing."""
    padded = (seq_len + 31) // 32 * 32
    shape = (1, padded, heads, 128)
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    k.sub_(k[:, :seq_len].mean(dim=1, keepdim=True))
    for tensor in (q, k, v):
        tensor[:, seq_len:] = 0

    def quantize_int8(tensor, multiplier=1.0):
        blocks = tensor.float().reshape(1, padded // 32, 32, heads, 128)
        blocks.mul_(multiplier)
        scale = blocks.abs().amax(dim=(2, 4), keepdim=True) / 127
        values = blocks / scale
        values = (values + 0.5 * values.sign()).to(torch.int8)
        scales = scale.reshape(1, padded // 32, heads).transpose(1, 2).contiguous()
        return values.reshape(shape), scales

    q_int8, q_scale = quantize_int8(q, math.log2(math.e) / math.sqrt(128))
    k_int8, k_scale = quantize_int8(k)
    v_scale = v.abs().amax(dim=1).float() / torch.finfo(torch.float8_e4m3fn).max
    v_fp8 = (v.float() / v_scale[:, None]).to(torch.float8_e4m3fn)
    return (
        q_int8,
        k_int8,
        v_fp8.permute(0, 2, 3, 1).contiguous(),
        q_scale,
        k_scale,
        v_scale,
    )


@torch.inference_mode()
def benchmark(original, changed, sp, repeats, seq_len=SEQ_LEN):
    heads = 56 // sp
    padded = (seq_len + 31) // 32 * 32
    inputs = make_inputs(heads, seq_len)
    # The production MiniMax-H3 B1/D128 configuration for a partial key tail.
    config = {
        "num_heads": heads,
        "block_m": 128,
        "block_n": 32,
        "lds_padding": 8,
        "kv_prefetch_mode": "kv",
        "gated_o_rescale": True,
        "experimental_lds_reserve_bytes": 32768,
        "experimental_key_tail_peel_seq_len": seq_len,
    }
    kernels = [original(**config), changed(**config)]
    outputs = [
        torch.empty((1, padded, heads, 128), device="cuda", dtype=torch.bfloat16)
        for _ in kernels
    ]

    def launch(index):
        kernels[index](
            *inputs,
            outputs[index],
            1,
            padded,
            seq_len,
            stream=torch.cuda.current_stream(),
        )

    for index in range(2):
        for _ in range(2):
            launch(index)
    torch.cuda.synchronize()
    reference = outputs[0][:, :seq_len].clone()
    if not bool(torch.isfinite(reference).all()):
        raise RuntimeError("The unchanged kernel produced nonfinite output")

    times = [[], []]
    for repeat in range(repeats):
        # Alternate which kernel runs first to reduce ordering bias.
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            launch(index)
            end.record()
            end.synchronize()
            times[index].append(start.elapsed_time(end))
            output = outputs[index][:, :seq_len].contiguous()
            if not bool(torch.isfinite(output).all()) or not torch.equal(
                reference.view(torch.uint8), output.view(torch.uint8)
            ):
                raise RuntimeError(f"Output mismatch: SP{sp}, kernel {index}")

    baseline_ms, changed_ms = map(statistics.median, times)
    print(
        f"SP{sp:<2} {heads:>5} {baseline_ms:>14.3f} {changed_ms:>14.3f} "
        f"{baseline_ms / changed_ms:>8.3f}x  byte-identical",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sp", nargs="+", type=int, choices=(2, 4, 8), default=[2, 4, 8]
    )
    parser.add_argument("--repeats", type=int, default=5, help="timed calls per kernel")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available() or not torch.version.hip:
        parser.error("this benchmark requires a ROCm gfx1201 GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        parser.error(f"this kernel requires gfx1201; found {arch}")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    print(f"GPU: {arch}; torch: {torch.__version__}; FlyDSL: {flydsl.__version__}")
    print(f"Unchanged: {BASELINE}; changed: {ROOT / KERNEL}")
    print(f"S={SEQ_LEN}, D=128, B=1; core-only GPU median; {args.repeats} calls/kernel")
    print(
        "SP   heads   unchanged_ms     changed_ms   speedup  output check", flush=True
    )
    with tempfile.TemporaryDirectory(prefix="sage-baseline-") as tmp:
        source = subprocess.run(
            ["git", "show", f"{BASELINE}:{KERNEL.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        baseline_path = Path(tmp) / "sage_baseline.py"
        baseline_path.write_bytes(source)
        original = load_builder(baseline_path, "sage_baseline")
        changed = load_builder(ROOT / KERNEL, "sage_changed")
        for sp in args.sp:
            benchmark(original, changed, sp, args.repeats)


if __name__ == "__main__":
    main()
