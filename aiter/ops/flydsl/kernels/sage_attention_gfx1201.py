# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native RDNA4 building blocks for SageAttention2 on gfx1201.

The public attention kernel is added incrementally.  This module starts with
small ABI probes for the two matrix operations that define the V2 compute
path.  They are also useful as numerical tests on real hardware because their
inputs and outputs are the packed per-lane WMMA fragments, with no attention
or quantization logic around them.

gfx1201 wave32 fragment ABI for one 16x16x16 operation:

* INT8/FP8 A and B: two packed i32 VGPRs per lane (eight elements);
* INT8 result: eight i32 VGPRs per lane;
* FP8 result: eight f32 VGPRs per lane.

The probes intentionally use the ROCDL operations instead of a generic dot so
compilation must either produce the required native ISA or fail.
"""

from __future__ import annotations

import argparse
import math as host_math
import os
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


GFX1201_WAVE_SIZE = 32
GFX1201_WMMA_M = 16
GFX1201_WMMA_N = 16
GFX1201_WMMA_K = 16
GFX1201_WMMA_INPUT_I32S_PER_LANE = 2
GFX1201_WMMA_ACCUMULATORS_PER_LANE = 8

_QK_PROBE_NAME = "sage_gfx1201_qk_i8_wmma_16x16x16"
_PV_PROBE_NAME = "sage_gfx1201_pv_fp8_wmma_16x16x16"
_QK_ISA = "v_wmma_i32_16x16x16_iu8"
_PV_ISA = "v_wmma_f32_16x16x16_fp8_fp8"
_LOG2E = host_math.log2(host_math.e)
_FP8_P_OFFSET = 8.807


def _raw(value):
    if isinstance(value, ir.Value):
        return value
    if hasattr(value, "ir_value"):
        return _raw(value.ir_value())
    return ir.Value._CAPICreate(value._CAPIPtr)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _pointer_to_llvm_ptr(ptr) -> ir.Value:
    ptr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
    return _llvm.IntToPtrOp(_llvm_ptr_ty(), ptr_i64).result


def _load_vector(ptr, element_index, elem_type, vector_type):
    address = buffer_ops.get_element_ptr(
        ptr,
        fx.Int64(element_index),
        elem_type=elem_type,
    )
    return _llvm.LoadOp(vector_type, _raw(address)).result


def _store_vector(ptr, element_index, elem_type, value):
    address = buffer_ops.get_element_ptr(
        ptr,
        fx.Int64(element_index),
        elem_type=elem_type,
    )
    _llvm.StoreOp(_raw(value), _raw(address))


def _set_waves_per_eu(waves_per_eu: int):
    ctx = CompilationContext.get_current()
    for op in ctx.gpu_module_body.operations:
        if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
            op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                T.i32,
                int(waves_per_eu),
            )


def build_sage_wmma_qk_probe(waves_per_eu: int = 2):
    """Build a signed INT8 16x16x16 WMMA packed-fragment probe."""

    @flyc.kernel(name=_QK_PROBE_NAME, known_block_size=[GFX1201_WAVE_SIZE, 1, 1])
    def qk_probe(a: fx.Pointer, b: fx.Pointer, d: fx.Pointer):
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        lane = fx.Index(gpu.thread_idx.x)
        a_ptr = _pointer_to_llvm_ptr(a)
        b_ptr = _pointer_to_llvm_ptr(b)
        d_ptr = _pointer_to_llvm_ptr(d)

        input_offset = lane * GFX1201_WMMA_INPUT_I32S_PER_LANE
        output_offset = lane * GFX1201_WMMA_ACCUMULATORS_PER_LANE
        a_frag = _load_vector(a_ptr, input_offset, T.i32, v2i32_type)
        b_frag = _load_vector(b_ptr, input_offset, T.i32, v2i32_type)
        accum = Vec.from_elements([fx.Int32(0)] * 8, fx.Int32).ir_value()

        result = rocdl.wmma_i32_16x16x16_iu8(
            v8i32_type,
            a_frag,
            b_frag,
            accum,
            signA=True,
            signB=True,
            clamp=False,
        ).result
        _store_vector(d_ptr, output_offset, T.i32, result)

    @flyc.jit
    def launch_qk_probe(
        a: fx.Pointer,
        b: fx.Pointer,
        d: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = qk_probe(a, b, d)
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(1, 1, 1),
            block=(GFX1201_WAVE_SIZE, 1, 1),
            stream=stream,
        )

    return launch_qk_probe


def build_sage_wmma_pv_probe(waves_per_eu: int = 2):
    """Build an E4M3 FP8 16x16x16 WMMA packed-fragment probe."""

    @flyc.kernel(name=_PV_PROBE_NAME, known_block_size=[GFX1201_WAVE_SIZE, 1, 1])
    def pv_probe(a: fx.Pointer, b: fx.Pointer, d: fx.Pointer):
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8f32_type = Vec.make_type(8, fx.Float32)
        lane = fx.Index(gpu.thread_idx.x)
        a_ptr = _pointer_to_llvm_ptr(a)
        b_ptr = _pointer_to_llvm_ptr(b)
        d_ptr = _pointer_to_llvm_ptr(d)

        input_offset = lane * GFX1201_WMMA_INPUT_I32S_PER_LANE
        output_offset = lane * GFX1201_WMMA_ACCUMULATORS_PER_LANE
        a_frag = _load_vector(a_ptr, input_offset, T.i32, v2i32_type)
        b_frag = _load_vector(b_ptr, input_offset, T.i32, v2i32_type)
        accum = Vec.from_elements([fx.Float32(0.0)] * 8, fx.Float32).ir_value()

        result = rocdl.wmma_f32_16x16x16_fp8_fp8(
            v8f32_type,
            a_frag,
            b_frag,
            accum,
        ).result
        _store_vector(d_ptr, output_offset, T.f32, result)

    @flyc.jit
    def launch_pv_probe(
        a: fx.Pointer,
        b: fx.Pointer,
        d: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = pv_probe(a, b, d)
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(1, 1, 1),
            block=(GFX1201_WAVE_SIZE, 1, 1),
            stream=stream,
        )

    return launch_pv_probe


def build_sage_wmma_rate_probe(
    kind: str,
    *,
    iterations: int = 512,
    independent_accumulators: int = 8,
    block_size: int = 256,
    grid_blocks: int = 1200,
    waves_per_eu: int = 2,
):
    """Build a saturated, independently accumulated WMMA throughput probe."""

    kind = str(kind).lower()
    if kind not in ("qk", "pv"):
        raise ValueError("Sage WMMA rate probe kind must be qk or pv")
    if iterations < 1 or iterations > 1024:
        raise ValueError("Sage WMMA rate probe iterations must be in [1, 1024]")
    if independent_accumulators not in (1, 2, 4, 8):
        raise ValueError(
            "Sage WMMA rate probe accumulator count must be 1, 2, 4 or 8"
        )
    if block_size < GFX1201_WAVE_SIZE or block_size % GFX1201_WAVE_SIZE:
        raise ValueError("Sage WMMA rate probe block size must be wave-aligned")
    if grid_blocks < 1:
        raise ValueError("Sage WMMA rate probe grid must be positive")

    name = (
        f"sage_gfx1201_wmma_rate_{kind}_i{iterations}"
        f"a{independent_accumulators}_b{block_size}_g{grid_blocks}"
    )

    @flyc.kernel(name=name, known_block_size=[block_size, 1, 1])
    def rate_probe(a: fx.Pointer, b: fx.Pointer, d: fx.Pointer):
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        v8f32_type = Vec.make_type(8, fx.Float32)
        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % GFX1201_WAVE_SIZE
        global_tid = fx.Index(gpu.block_idx.x) * block_size + tid
        a_ptr = _pointer_to_llvm_ptr(a)
        b_ptr = _pointer_to_llvm_ptr(b)
        d_ptr = _pointer_to_llvm_ptr(d)

        input_offset = lane * GFX1201_WMMA_INPUT_I32S_PER_LANE
        output_offset = global_tid * GFX1201_WMMA_ACCUMULATORS_PER_LANE
        a_frag = _load_vector(a_ptr, input_offset, T.i32, v2i32_type)
        b_frag = _load_vector(b_ptr, input_offset, T.i32, v2i32_type)
        if const_expr(kind == "qk"):
            accumulators = [
                Vec.from_elements([fx.Int32(0)] * 8, fx.Int32).ir_value()
                for _ in range_constexpr(independent_accumulators)
            ]
        else:
            accumulators = [
                Vec.from_elements([fx.Float32(0.0)] * 8, fx.Float32).ir_value()
                for _ in range_constexpr(independent_accumulators)
            ]

        for iteration in range_constexpr(iterations):
            stream = iteration % independent_accumulators
            if const_expr(kind == "qk"):
                accumulators[stream] = rocdl.wmma_i32_16x16x16_iu8(
                    v8i32_type,
                    a_frag,
                    b_frag,
                    accumulators[stream],
                    signA=True,
                    signB=True,
                    clamp=False,
                ).result
            else:
                accumulators[stream] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                    v8f32_type,
                    a_frag,
                    b_frag,
                    accumulators[stream],
                ).result

        combined = accumulators[0]
        for stream in range_constexpr(1, independent_accumulators):
            if const_expr(kind == "qk"):
                combined = arith.addi(_raw(combined), _raw(accumulators[stream]))
            else:
                combined = arith.addf(_raw(combined), _raw(accumulators[stream]))
        _store_vector(
            d_ptr,
            output_offset,
            T.i32 if kind == "qk" else T.f32,
            combined,
        )

    @flyc.jit
    def launch_rate_probe(
        a: fx.Pointer,
        b: fx.Pointer,
        d: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rate_probe(a, b, d)
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(grid_blocks, 1, 1),
            block=(block_size, 1, 1),
            stream=stream,
        )

    launch_rate_probe.probe_metadata = {
        "kind": kind,
        "kernel_name": name,
        "iterations": iterations,
        "independent_accumulators": independent_accumulators,
        "block_size": block_size,
        "grid_blocks": grid_blocks,
        "waves": grid_blocks * block_size // GFX1201_WAVE_SIZE,
        "wmma_instructions": (
            grid_blocks
            * block_size
            // GFX1201_WAVE_SIZE
            * iterations
        ),
    }
    return launch_rate_probe


def build_sage_lds_health_probe(
    conflict: bool,
    *,
    iterations: int = 64,
    grid_blocks: int = 120,
    waves_per_eu: int = 2,
):
    """Build a bank-free or forced-same-bank LDS counter health probe."""

    if iterations < 1 or iterations > 64:
        raise ValueError("Sage LDS health iterations must be in [1, 64]")
    if grid_blocks < 1:
        raise ValueError("Sage LDS health grid must be positive")
    name = (
        "sage_gfx1201_lds_conflict_health"
        if conflict
        else "sage_gfx1201_lds_free_health"
    )
    lds_bytes = GFX1201_WAVE_SIZE * iterations * 8
    allocator = SmemAllocator(
        None,
        arch=os.environ.get("FLYDSL_GPU_ARCH", "gfx1201"),
        global_sym_name=f"{name}_smem",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + lds_bytes

    @flyc.kernel(name=name, known_block_size=[GFX1201_WAVE_SIZE, 1, 1])
    def lds_probe(d: fx.Pointer):
        v2i32_type = Vec.make_type(2, fx.Int32)
        base_ptr = allocator.get_base()
        lds = SmemPtr(
            base_ptr,
            lds_offset,
            T.i8,
            shape=(lds_bytes,),
        ).get()
        lane = fx.Index(gpu.thread_idx.x)
        global_tid = fx.Index(gpu.block_idx.x) * GFX1201_WAVE_SIZE + lane
        d_ptr = _pointer_to_llvm_ptr(d)
        seed = Vec.from_elements([fx.Int32(1), fx.Int32(2)], fx.Int32).ir_value()
        accum = Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32).ir_value()

        for iteration in range_constexpr(iterations):
            if const_expr(conflict):
                address = lane * iterations * 8 + iteration * 8
            else:
                address = (iteration * GFX1201_WAVE_SIZE + lane) * 8
            Vec(seed).store(lds, [address])
        gpu.barrier()
        for iteration in range_constexpr(iterations):
            if const_expr(conflict):
                address = lane * iterations * 8 + iteration * 8
            else:
                address = (iteration * GFX1201_WAVE_SIZE + lane) * 8
            value = Vec.load(v2i32_type, lds, [address])
            accum = arith.addi(_raw(accum), _raw(value))
        _store_vector(d_ptr, global_tid * 2, T.i32, accum)

    @flyc.jit
    def launch_lds_probe(
        d: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        launcher = lds_probe(d)
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(grid_blocks, 1, 1),
            block=(GFX1201_WAVE_SIZE, 1, 1),
            stream=stream,
        )

    launch_lds_probe.probe_metadata = {
        "conflict": bool(conflict),
        "kernel_name": name,
        "iterations": iterations,
        "grid_blocks": grid_blocks,
        "lds_bytes": lds_bytes,
        "lds_loads": grid_blocks * iterations,
    }
    return launch_lds_probe


def build_sage_attention_v2_core(
    num_heads: int,
    head_dim: int = 128,
    output_dtype: str = "bf16",
    block_m: int = 128,
    block_n: int = 32,
    waves_per_eu: int = 2,
    lds_padding: int = 16,
    k_lds_padding: int | None = None,
    v_lds_padding: int | None = None,
    pre_load_v: bool = False,
    kv_prefetch_mode: str | None = None,
    use_fp8_p_offset: bool = True,
    no_key_tail: bool = False,
    gated_o_rescale: bool = False,
    experimental_barrier_overlap: bool = False,
    experimental_lds_reserve_bytes: int = 0,
    experimental_rows_per_wave: int = 16,
    experimental_softmax_sum_parts: int = 1,
    experimental_prefetch_stage: str = "early",
    experimental_key_tail_peel_seq_len: int = 0,
    experimental_grid_blocks: int = 0,
    return_lse: bool = False,
):
    """Build the pre-quantized, non-causal gfx1201 SageAttention2 core.

    Q and K are sequence-major signed INT8. Q scales cover 32 query rows and
    already include ``softmax_scale * log2(e)``; K scales cover ``block_n``
    rows. V is E4M3 FP8 in RDNA-native ``[B, H, D, padded_S]`` order with one
    FP32 scale for every ``[B, H, D]`` channel. Output is BSHD BF16/FP16.

    Storage sequence length must be padded to ``block_n`` by the host wrapper.
    The separate valid sequence length masks padded keys out of the online
    softmax. The first production path deliberately handles self-attention
    only; asymmetric lengths and GQA stay on the existing fallback until
    separately validated.
    """

    if head_dim != 128:
        raise ValueError("gfx1201 SageAttention2 currently requires head_dim=128")
    if output_dtype not in ("bf16", "f16"):
        raise ValueError(f"unsupported Sage output dtype: {output_dtype}")
    if block_m not in (64, 128, 256):
        raise ValueError("Sage block_m must be 64, 128 or 256")
    if block_n not in (32, 64):
        raise ValueError("Sage block_n must be 32 or 64")
    if lds_padding not in (4, 8, 16):
        raise ValueError("Sage lds_padding must be 4, 8, or 16")
    split_padding_values = (0, 4, 8, 12, 16, 20, 24, 28, 32)
    if k_lds_padding is not None and k_lds_padding not in split_padding_values:
        raise ValueError(
            "Sage K LDS padding must be a 4-byte multiple in [0, 32]"
        )
    if v_lds_padding is not None and v_lds_padding not in split_padding_values:
        raise ValueError(
            "Sage V LDS padding must be a 4-byte multiple in [0, 32]"
        )
    if kv_prefetch_mode is None:
        kv_prefetch_mode = "k" if (block_m, block_n) == (128, 32) else "none"
    kv_prefetch_mode = str(kv_prefetch_mode).lower()
    if kv_prefetch_mode not in ("none", "v", "k", "kv", "e1", "e12"):
        raise ValueError(
            "Sage KV prefetch mode must be none, v, k, kv, e1 or e12"
        )
    if kv_prefetch_mode in ("e1", "e12") and (
        block_m != 128 or block_n != 32 or pre_load_v
    ):
        raise ValueError("Sage E1/E12 requires BM128, BN32 and PRE_LOAD_V=false")
    if kv_prefetch_mode != "none" and (
        block_m not in (64, 128) or block_n != 32
    ):
        raise ValueError("Sage KV prefetch requires BM64/128 and BN32")
    if experimental_barrier_overlap and (
        block_m != 128
        or block_n != 32
        or pre_load_v
        or kv_prefetch_mode not in ("k", "v", "kv")
    ):
        raise ValueError(
            "Sage experimental barrier overlap requires BM128, BN32, "
            "PRE_LOAD_V=false and K/V/KV prefetch"
        )
    if (
        experimental_lds_reserve_bytes < 0
        or experimental_lds_reserve_bytes > 65536
        or experimental_lds_reserve_bytes % 256
    ):
        raise ValueError(
            "Sage experimental LDS reservation must be a 256-byte multiple "
            "in [0, 65536]"
        )
    if experimental_rows_per_wave not in (16, 32):
        raise ValueError("Sage experimental rows per wave must be 16 or 32")
    if experimental_rows_per_wave == 32 and (
        block_m != 128
        or block_n != 32
        or pre_load_v
        or kv_prefetch_mode not in ("none", "v", "k", "kv")
    ):
        raise ValueError(
            "Sage 32 rows/wave requires BM128, BN32, PRE_LOAD_V=false, "
            "and prefetch none/v/k/kv"
        )
    if experimental_softmax_sum_parts not in (1, 2, 4):
        raise ValueError("Sage experimental softmax sum parts must be 1, 2 or 4")
    experimental_prefetch_stage = str(experimental_prefetch_stage).lower()
    if experimental_prefetch_stage not in ("early", "after_qk", "before_pv"):
        raise ValueError(
            "Sage experimental prefetch stage must be early, after_qk or before_pv"
        )
    if experimental_prefetch_stage != "early" and (
        block_m != 128
        or block_n != 32
        or pre_load_v
        or kv_prefetch_mode not in ("v", "k", "kv")
    ):
        raise ValueError(
            "Sage delayed prefetch requires BM128, BN32, PRE_LOAD_V=false "
            "and K/V/KV prefetch"
        )
    if experimental_key_tail_peel_seq_len < 0:
        raise ValueError(
            "Sage experimental key-tail peel sequence length must be non-negative"
        )
    if experimental_key_tail_peel_seq_len:
        if experimental_key_tail_peel_seq_len % block_n == 0:
            raise ValueError(
                "Sage experimental key-tail peel requires a sequence length "
                "with a partial block_n tail"
            )
        if (
            block_m != 128
            or block_n != 32
            or pre_load_v
            or kv_prefetch_mode != "kv"
            or experimental_rows_per_wave not in (16, 32)
            or experimental_prefetch_stage != "early"
            or no_key_tail
        ):
            raise ValueError(
                "Sage experimental key-tail peel requires BM128, BN32, "
                "PRE_LOAD_V=false, KV prefetch, 16/32 rows/wave, early "
                "prefetch and NO_KEY_TAIL=false"
            )
    if experimental_grid_blocks < 0:
        raise ValueError("Sage experimental grid blocks must be non-negative")

    WARP_SIZE = GFX1201_WAVE_SIZE
    WMMA_K = GFX1201_WMMA_K
    WMMA_ROWS = GFX1201_WMMA_M
    ROWS_PER_WAVE = int(experimental_rows_per_wave)
    ROW_GROUPS_PER_WAVE = ROWS_PER_WAVE // WMMA_ROWS
    WMMA_LANE_K = 8
    K_SUB_N = 32
    VEC_WIDTH = 16
    Q_SCALE_ROWS = 32

    BLOCK_M = int(block_m)
    BLOCK_N = int(block_n)
    PRE_LOAD_V = bool(pre_load_v)
    DB_KV_PIPELINE = kv_prefetch_mode in ("e1", "e12")
    S_LOOKAHEAD = kv_prefetch_mode == "e12"
    PREFETCH_K = kv_prefetch_mode in ("k", "kv", "e1", "e12")
    PREFETCH_V = kv_prefetch_mode in ("v", "kv", "e1", "e12")
    USE_FP8_P_OFFSET = bool(use_fp8_p_offset)
    NO_KEY_TAIL = bool(no_key_tail)
    GATED_O_RESCALE = bool(gated_o_rescale)
    EXPERIMENTAL_BARRIER_OVERLAP = bool(experimental_barrier_overlap)
    EXPERIMENTAL_LDS_RESERVE_BYTES = int(experimental_lds_reserve_bytes)
    SOFTMAX_SUM_PARTS = int(experimental_softmax_sum_parts)
    PREFETCH_STAGE = experimental_prefetch_stage
    KEY_TAIL_PEEL_SEQ_LEN = int(experimental_key_tail_peel_seq_len)
    EXPERIMENTAL_GRID_BLOCKS = int(experimental_grid_blocks)
    RETURN_LSE = bool(return_lse)
    BLOCK_SIZE = (BLOCK_M // ROWS_PER_WAVE) * WARP_SIZE
    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8
    K_STEPS_QK = head_dim // WMMA_K
    D_CHUNKS = head_dim // GFX1201_WMMA_N
    PV_K_STEPS = K_SUB_N // WMMA_K

    K_LDS_PADDING = (
        lds_padding if k_lds_padding is None else int(k_lds_padding)
    )
    V_LDS_PADDING = (
        lds_padding if v_lds_padding is None else int(v_lds_padding)
    )
    K_STRIDE = head_dim + K_LDS_PADDING
    V_STRIDE = BLOCK_N + V_LDS_PADDING
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = head_dim * V_STRIDE
    LDS_BUFFER_COUNT = 2 if DB_KV_PIPELINE else 1
    LDS_V_BASE = LDS_BUFFER_COUNT * LDS_K_TILE_SIZE
    LDS_TOTAL_BYTES = LDS_BUFFER_COUNT * (
        LDS_K_TILE_SIZE + LDS_V_TILE_SIZE
    )

    K_THREADS_PER_ROW = head_dim // VEC_WIDTH
    K_LOAD_ITEMS = BLOCK_N * K_THREADS_PER_ROW
    K_LOAD_BATCHES = (K_LOAD_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
    V_CHUNKS_PER_ROW = BLOCK_N // VEC_WIDTH
    V_LOAD_ITEMS = head_dim * V_CHUNKS_PER_ROW
    V_LOAD_BATCHES = (V_LOAD_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
    ROW_STATE_WIDTH = 2 + D_CHUNKS
    S_LOOKAHEAD_STATE_BASE = ROW_GROUPS_PER_WAVE * ROW_STATE_WIDTH
    K_PREFETCH_STATE_BASE = S_LOOKAHEAD_STATE_BASE + (
        ROW_GROUPS_PER_WAVE * NUM_S_ACCS if S_LOOKAHEAD else 0
    )
    V_PREFETCH_STATE_BASE = K_PREFETCH_STATE_BASE + (
        K_LOAD_BATCHES if PREFETCH_K else 0
    )

    gpu_arch = os.environ.get("FLYDSL_GPU_ARCH", "gfx1201")
    path_tag = (
        f"M{BLOCK_M}N{BLOCK_N}P{lds_padding}W{waves_per_eu}H{num_heads}"
        f"V{int(PRE_LOAD_V)}G{kv_prefetch_mode}O{int(USE_FP8_P_OFFSET)}"
        f"L{int(RETURN_LSE)}"
    )
    if k_lds_padding is not None or v_lds_padding is not None:
        path_tag += f"PK{K_LDS_PADDING}PV{V_LDS_PADDING}"
    if NO_KEY_TAIL or GATED_O_RESCALE:
        path_tag += f"T{int(NO_KEY_TAIL)}R{int(GATED_O_RESCALE)}"
    if EXPERIMENTAL_BARRIER_OVERLAP or EXPERIMENTAL_LDS_RESERVE_BYTES:
        path_tag += (
            f"B{int(EXPERIMENTAL_BARRIER_OVERLAP)}"
            f"X{EXPERIMENTAL_LDS_RESERVE_BYTES}"
        )
    if ROWS_PER_WAVE != GFX1201_WMMA_M:
        path_tag += f"Y{ROWS_PER_WAVE}"
    if SOFTMAX_SUM_PARTS != 1:
        path_tag += f"S{SOFTMAX_SUM_PARTS}"
    if PREFETCH_STAGE != "early":
        path_tag += f"F{'q' if PREFETCH_STAGE == 'after_qk' else 'p'}"
    if KEY_TAIL_PEEL_SEQ_LEN:
        path_tag += f"J{KEY_TAIL_PEEL_SEQ_LEN}"
    if EXPERIMENTAL_GRID_BLOCKS:
        path_tag += f"C{EXPERIMENTAL_GRID_BLOCKS}"
    path_tag += "HM"
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"sage_attention_v2_gfx1201_smem_{path_tag}",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + max(
        LDS_TOTAL_BYTES,
        EXPERIMENTAL_LDS_RESERVE_BYTES,
    )

    output_numeric = fx.BFloat16 if output_dtype == "bf16" else fx.Float16

    @flyc.kernel(
        name=f"sage_attention_v2_gfx1201_{path_tag}",
        known_block_size=[BLOCK_SIZE, 1, 1],
    )
    def sage_attention_core(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        QScale: fx.Pointer,
        KScale: fx.Pointer,
        VScale: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        padded_seq_len: fx.Int32,
        valid_seq_len: fx.Int32,
    ):
        fm_fast = arith.FastMathFlags.fast
        v8i8_type = Vec.make_type(8, fx.Int8)
        v16i8_type = Vec.make_type(16, fx.Int8)
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        v8f32_type = Vec.make_type(8, fx.Float32)
        v8out_type = Vec.make_type(8, output_numeric)

        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        qs_ptr = _pointer_to_llvm_ptr(QScale)
        ks_ptr = _pointer_to_llvm_ptr(KScale)
        vs_ptr = _pointer_to_llvm_ptr(VScale)
        o_ptr = _pointer_to_llvm_ptr(O)
        lse_ptr = _pointer_to_llvm_ptr(LSE)

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _schedule_barrier():
            """Prevent the machine scheduler from crossing an experiment fence."""

            mask = arith.constant(0, type=T.i32)
            _llvm.call_intrinsic(
                None,
                "llvm.amdgcn.sched.barrier",
                [_raw(mask)],
                [],
                [],
            )

        def _global_load(ptr, index, elem_type, result_type):
            address = buffer_ops.get_element_ptr(
                ptr,
                fx.Int64(index),
                elem_type=elem_type,
            )
            return _llvm.LoadOp(result_type, _raw(address)).result

        def _global_store(ptr, index, elem_type, value):
            address = buffer_ops.get_element_ptr(
                ptr,
                fx.Int64(index),
                elem_type=elem_type,
            )
            _llvm.StoreOp(_raw(value), _raw(address))

        def _pack_i8_fragment(v8):
            return Vec(v8).bitcast(fx.Int32).ir_value()

        def _pack_fp8_probability(values):
            zero_i32 = arith.constant(0, type=T.i32)
            p0 = rocdl.cvt_pk_fp8_f32(
                T.i32, values[0], values[1], zero_i32, 0
            )
            p0 = rocdl.cvt_pk_fp8_f32(T.i32, values[2], values[3], p0, 1)
            p1 = rocdl.cvt_pk_fp8_f32(
                T.i32, values[4], values[5], zero_i32, 0
            )
            p1 = rocdl.cvt_pk_fp8_f32(T.i32, values[6], values[7], p1, 1)
            return Vec.from_elements([p0, p1], fx.Int32).ir_value()

        def _wmma_qk(a, b, accum):
            return rocdl.wmma_i32_16x16x16_iu8(
                v8i32_type,
                a,
                b,
                accum,
                signA=True,
                signB=True,
                clamp=False,
            ).result

        def _wmma_pv(a, b, accum):
            return rocdl.wmma_f32_16x16x16_fp8_fp8(
                v8f32_type,
                a,
                b,
                accum,
            ).result

        seq = fx.Index(padded_seq_len)
        valid_seq = fx.Index(valid_seq_len)
        base_ptr = allocator.get_base()
        lds = SmemPtr(
            base_ptr,
            lds_offset,
            T.i8,
            shape=(LDS_TOTAL_BYTES,),
        ).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        q_tiles = (valid_seq + BLOCK_M - 1) // BLOCK_M
        # Consecutive block IDs cover query tiles of one head for K/V reuse.
        head_idx = (block_id // q_tiles) % num_heads
        q_tile_idx = block_id % q_tiles
        batch_idx = block_id // (q_tiles * num_heads)
        q_start = q_tile_idx * BLOCK_M
        q_rows = [
            q_start + wave_id * ROWS_PER_WAVE + row_group * WMMA_ROWS + lane16
            for row_group in range_constexpr(ROW_GROUPS_PER_WAVE)
        ]

        def qk_global_index(token, col):
            return ((batch_idx * seq + token) * num_heads + head_idx) * head_dim + col

        def v_global_index(d, token):
            return ((batch_idx * num_heads + head_idx) * head_dim + d) * seq + token

        def lse_global_index(token):
            return (batch_idx * num_heads + head_idx) * seq + token

        q_in_bounds = [
            arith.cmpi(
                arith.CmpIPredicate.slt,
                _raw(q_rows[row_group]),
                _raw(valid_seq),
            )
            for row_group in range_constexpr(ROW_GROUPS_PER_WAVE)
        ]
        q_rows_safe = [
            fx.Index(
                ArithValue(q_in_bounds[row_group]).select(
                    q_rows[row_group], fx.Index(0)
                )
            )
            for row_group in range_constexpr(ROW_GROUPS_PER_WAVE)
        ]
        zero_v8i8 = Vec.filled(8, 0, fx.Int8).ir_value()
        q_fragments = []
        for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
            row_fragments = []
            for ks in range_constexpr(K_STEPS_QK):
                q_col = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                q_raw = _global_load(
                    q_ptr,
                    qk_global_index(q_rows_safe[row_group], q_col),
                    T.i8,
                    v8i8_type,
                )
                q_safe = ArithValue(q_in_bounds[row_group]).select(
                    q_raw, zero_v8i8
                )
                row_fragments.append(_pack_i8_fragment(q_safe))
            q_fragments.append(row_fragments)

        q_scale_blocks = seq // Q_SCALE_ROWS
        q_scales = []
        for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
            if const_expr(ROWS_PER_WAVE == 32 and row_group == 1):
                # Quantization uses one Q scale per 32 rows.  The two WMMA
                # row groups owned by an R32 wave therefore share it.
                q_scales.append(q_scales[0])
            else:
                q_scale_index = (
                    (batch_idx * num_heads + head_idx) * q_scale_blocks
                    + q_rows_safe[row_group] // Q_SCALE_ROWS
                )
                q_scales.append(
                    _global_load(qs_ptr, q_scale_index, T.f32, T.f32)
                )

        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_ln2 = fx.Float32(host_math.log(2.0))
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32).ir_value()
        c_zero_v8i32 = Vec.filled(8, 0, fx.Int32).ir_value()
        c_neg_fp8_p_offset = fx.Float32(-_FP8_P_OFFSET)
        width_i32 = fx.Int32(WARP_SIZE)
        peer_i32 = fx.Int32(16)

        def reduction_peer(value):
            return fx.Float32(value).shuffle_xor(peer_i32, width_i32)

        def _load_k_tile_global(tile_start):
            fragments = []
            for load_batch in range_constexpr(K_LOAD_BATCHES):
                k_item = tid + load_batch * BLOCK_SIZE
                k_row = k_item // K_THREADS_PER_ROW
                k_col = (k_item % K_THREADS_PER_ROW) * VEC_WIDTH
                fragments.append(
                    _global_load(
                        k_ptr,
                        qk_global_index(tile_start + k_row, k_col),
                        T.i8,
                        v16i8_type,
                    )
                )
            return fragments

        def _store_k_tile_lds(fragments, buffer_index):
            for load_batch in range_constexpr(K_LOAD_BATCHES):
                k_item = tid + load_batch * BLOCK_SIZE
                k_row = k_item // K_THREADS_PER_ROW
                k_col = (k_item % K_THREADS_PER_ROW) * VEC_WIDTH
                Vec(fragments[load_batch]).store(
                    lds,
                    [
                        buffer_index * LDS_K_TILE_SIZE
                        + k_row * K_STRIDE
                        + k_col
                    ],
                )

        def _load_v_tile_global(tile_start):
            fragments = []
            for load_batch in range_constexpr(V_LOAD_BATCHES):
                v_item = tid + load_batch * BLOCK_SIZE
                d_row = v_item // V_CHUNKS_PER_ROW
                token_offset = (v_item % V_CHUNKS_PER_ROW) * VEC_WIDTH
                fragments.append(
                    _global_load(
                        v_ptr,
                        v_global_index(d_row, tile_start + token_offset),
                        T.i8,
                        v16i8_type,
                    )
                )
            return fragments

        def _store_v_tile_lds(fragments, buffer_index):
            for load_batch in range_constexpr(V_LOAD_BATCHES):
                v_item = tid + load_batch * BLOCK_SIZE
                d_row = v_item // V_CHUNKS_PER_ROW
                token_offset = (v_item % V_CHUNKS_PER_ROW) * VEC_WIDTH
                Vec(fragments[load_batch]).store(
                    lds,
                    [
                        LDS_V_BASE
                        + buffer_index * LDS_V_TILE_SIZE
                        + d_row * V_STRIDE
                        + token_offset
                    ],
                )

        def _load_v_fragment(st_idx, pks, dc, buffer_index):
            d_pos = fx.Index(dc * GFX1201_WMMA_N) + lane16
            token_pos = (
                st_idx * K_SUB_N
                + pks * WMMA_K
                + klane * WMMA_LANE_K
            )
            return Vec.load(
                v8i8_type,
                lds,
                [
                    LDS_V_BASE
                    + buffer_index * LDS_V_TILE_SIZE
                    + d_pos * V_STRIDE
                    + token_pos
                ],
            )

        def _safe_tile_start(tile_start):
            in_bounds = arith.cmpi(
                arith.CmpIPredicate.slt,
                _raw(tile_start),
                _raw(seq),
            )
            return fx.Index(
                ArithValue(in_bounds).select(tile_start, fx.Index(0))
            )

        def _compute_qk(buffer_index):
            accumulators = [
                [c_zero_v8i32 for _ in range(NUM_S_ACCS)]
                for _ in range(ROW_GROUPS_PER_WAVE)
            ]
            for ks in range_constexpr(K_STEPS_QK):
                k_col = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base = st_idx * K_SUB_N
                    k_row_a = lane16 + st_base
                    k_row_b = lane16 + st_base + 16
                    k_buffer_base = buffer_index * LDS_K_TILE_SIZE
                    k_a = Vec.load(
                        v8i8_type,
                        lds,
                        [k_buffer_base + k_row_a * K_STRIDE + k_col],
                    )
                    k_b = Vec.load(
                        v8i8_type,
                        lds,
                        [k_buffer_base + k_row_b * K_STRIDE + k_col],
                    )
                    acc_a = st_idx * 2
                    acc_b = acc_a + 1
                    packed_k_a = _pack_i8_fragment(k_a)
                    packed_k_b = _pack_i8_fragment(k_b)
                    for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
                        accumulators[row_group][acc_a] = _wmma_qk(
                            packed_k_a,
                            q_fragments[row_group][ks],
                            accumulators[row_group][acc_a],
                        )
                        accumulators[row_group][acc_b] = _wmma_qk(
                            packed_k_b,
                            q_fragments[row_group][ks],
                            accumulators[row_group][acc_b],
                        )
            return accumulators

        zero_buffer = fx.Index(0)
        one_buffer = fx.Index(1)
        initial_s_lookahead = []
        initial_k_prefetch = []
        initial_v_prefetch = []
        if const_expr(S_LOOKAHEAD):
            initial_k = _load_k_tile_global(fx.Index(0))
            _store_k_tile_lds(initial_k, zero_buffer)
            initial_v = _load_v_tile_global(fx.Index(0))
            _store_v_tile_lds(initial_v, zero_buffer)

            k_one_start = _safe_tile_start(fx.Index(BLOCK_N))
            initial_k_one = _load_k_tile_global(k_one_start)
            _store_k_tile_lds(initial_k_one, one_buffer)

            initial_k_prefetch = _load_k_tile_global(
                _safe_tile_start(fx.Index(2 * BLOCK_N))
            )
            initial_v_prefetch = _load_v_tile_global(k_one_start)
            gpu.barrier()
            initial_s_lookahead = _compute_qk(zero_buffer)
        elif const_expr(DB_KV_PIPELINE):
            initial_k = _load_k_tile_global(fx.Index(0))
            _store_k_tile_lds(initial_k, zero_buffer)
            initial_v = _load_v_tile_global(fx.Index(0))
            _store_v_tile_lds(initial_v, zero_buffer)

            one_start = _safe_tile_start(fx.Index(BLOCK_N))
            initial_k_prefetch = _load_k_tile_global(one_start)
            initial_v_prefetch = _load_v_tile_global(one_start)
        else:
            if const_expr(PREFETCH_K):
                initial_k_prefetch = _load_k_tile_global(fx.Index(0))
            if const_expr(PREFETCH_V):
                initial_v_prefetch = _load_v_tile_global(fx.Index(0))

        init_args = []
        for _ in range_constexpr(ROW_GROUPS_PER_WAVE):
            init_args.extend([_raw(c_neg_inf), _raw(c_zero_f)])
            for _ in range_constexpr(D_CHUNKS):
                init_args.append(c_zero_v8f32)
        if const_expr(S_LOOKAHEAD):
            for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
                for s_acc in range_constexpr(NUM_S_ACCS):
                    init_args.append(initial_s_lookahead[row_group][s_acc])
        if const_expr(PREFETCH_K):
            for load_batch in range_constexpr(K_LOAD_BATCHES):
                init_args.append(initial_k_prefetch[load_batch])
        if const_expr(PREFETCH_V):
            for load_batch in range_constexpr(V_LOAD_BATCHES):
                init_args.append(initial_v_prefetch[load_batch])

        def _process_kv_tile(
            kv_start,
            inner,
            *,
            mask_key_tail,
            prefetch_next,
            carry_prefetch,
        ):
            row_states = []
            for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
                state_base = row_group * ROW_STATE_WIDTH
                row_states.append(
                    (
                        inner[state_base],
                        inner[state_base + 1],
                        [
                            inner[state_base + 2 + dc]
                            for dc in range_constexpr(D_CHUNKS)
                        ],
                    )
                )
            s_accs = []
            if const_expr(S_LOOKAHEAD):
                for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
                    s_base = (
                        S_LOOKAHEAD_STATE_BASE + row_group * NUM_S_ACCS
                    )
                    s_accs.append(
                        [
                            inner[s_base + s_acc]
                            for s_acc in range_constexpr(NUM_S_ACCS)
                        ]
                    )
            current_k_prefetch = []
            if const_expr(PREFETCH_K):
                current_k_prefetch = [
                    inner[K_PREFETCH_STATE_BASE + load_batch]
                    for load_batch in range_constexpr(K_LOAD_BATCHES)
                ]
            current_v_prefetch = []
            if const_expr(PREFETCH_V):
                current_v_prefetch = [
                    inner[V_PREFETCH_STATE_BASE + load_batch]
                    for load_batch in range_constexpr(V_LOAD_BATCHES)
                ]

            current_buffer = zero_buffer
            next_buffer = zero_buffer
            if const_expr(DB_KV_PIPELINE and prefetch_next):
                current_buffer = (kv_start // BLOCK_N) % 2
                next_buffer = one_buffer - current_buffer

                # One steady-state barrier protects both alternating buffers.
                gpu.barrier()
                k_store_buffer = (
                    current_buffer if S_LOOKAHEAD else next_buffer
                )
                _store_k_tile_lds(current_k_prefetch, k_store_buffer)
                _store_v_tile_lds(current_v_prefetch, next_buffer)
            else:
                if const_expr(PREFETCH_K):
                    _store_k_tile_lds(current_k_prefetch, zero_buffer)
                else:
                    for load_batch in range_constexpr(K_LOAD_BATCHES):
                        k_item = tid + load_batch * BLOCK_SIZE
                        if k_item < K_LOAD_ITEMS:
                            k_row = k_item // K_THREADS_PER_ROW
                            k_col = (k_item % K_THREADS_PER_ROW) * VEC_WIDTH
                            k_vec = _global_load(
                                k_ptr,
                                qk_global_index(kv_start + k_row, k_col),
                                T.i8,
                                v16i8_type,
                            )
                            Vec(k_vec).store(lds, [k_row * K_STRIDE + k_col])

                if const_expr(PREFETCH_V):
                    _store_v_tile_lds(current_v_prefetch, zero_buffer)
                else:
                    for load_batch in range_constexpr(V_LOAD_BATCHES):
                        v_item = tid + load_batch * BLOCK_SIZE
                        if v_item < V_LOAD_ITEMS:
                            d_row = v_item // V_CHUNKS_PER_ROW
                            token_offset = (v_item % V_CHUNKS_PER_ROW) * VEC_WIDTH
                            v_vec = _global_load(
                                v_ptr,
                                v_global_index(d_row, kv_start + token_offset),
                                T.i8,
                                v16i8_type,
                            )
                            Vec(v_vec).store(
                                lds,
                                [LDS_V_BASE + d_row * V_STRIDE + token_offset],
                            )

                gpu.barrier()

            # Keep next-tile VMEM values live while this tile computes, then
            # commit them to LDS at the next iteration.
            next_k_prefetch = []
            next_v_prefetch = []
            if const_expr(DB_KV_PIPELINE):
                k_prefetch_distance = 3 if S_LOOKAHEAD else 2
                next_k_prefetch = _load_k_tile_global(
                    _safe_tile_start(kv_start + k_prefetch_distance * BLOCK_N)
                )
                next_v_prefetch = _load_v_tile_global(
                    _safe_tile_start(kv_start + 2 * BLOCK_N)
                )
            elif const_expr(
                prefetch_next
                and (PREFETCH_K or PREFETCH_V)
                and PREFETCH_STAGE == "early"
            ):
                next_kv_start = kv_start + BLOCK_N
                safe_next_kv_start = _safe_tile_start(next_kv_start)
                if const_expr(PREFETCH_K):
                    next_k_prefetch = _load_k_tile_global(safe_next_kv_start)
                if const_expr(PREFETCH_V):
                    next_v_prefetch = _load_v_tile_global(safe_next_kv_start)

            next_s_lookahead = []
            if const_expr(S_LOOKAHEAD):
                # This QK chain is independent of the current softmax/PV.
                next_s_lookahead = _compute_qk(next_buffer)
            else:
                s_accs = _compute_qk(current_buffer)

            if const_expr(
                not DB_KV_PIPELINE
                and prefetch_next
                and (PREFETCH_K or PREFETCH_V)
                and PREFETCH_STAGE == "after_qk"
            ):
                next_kv_start = kv_start + BLOCK_N
                safe_next_kv_start = _safe_tile_start(next_kv_start)
                if const_expr(PREFETCH_K):
                    next_k_prefetch = _load_k_tile_global(safe_next_kv_start)
                if const_expr(PREFETCH_V):
                    next_v_prefetch = _load_v_tile_global(safe_next_kv_start)

            k_scale_blocks = seq // BLOCK_N
            k_scale_index = (
                (batch_idx * num_heads + head_idx) * k_scale_blocks
                + kv_start // BLOCK_N
            )
            k_scale = _global_load(ks_ptr, k_scale_index, T.f32, T.f32)
            row_next_states = []
            p_fragments_by_row = []
            for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
                m_running, l_running, o_accs = row_states[row_group]
                score_scale = _fmul(q_scales[row_group], k_scale)

                # Quantization scales are non-negative, so max can be reduced
                # in the raw score domain and scaled once per tile.
                raw_scores = []
                for st in range_constexpr(NUM_S_ACCS):
                    for item in range_constexpr(8):
                        score_f32 = arith.sitofp(
                            T.f32, Vec(s_accs[row_group][st])[item]
                        )
                        if const_expr(not mask_key_tail):
                            raw_scores.append(score_f32)
                        else:
                            key_offset = (
                                (st // 2) * K_SUB_N
                                + (st % 2) * WMMA_ROWS
                                + klane * WMMA_LANE_K
                                + item
                            )
                            key_in_bounds = arith.cmpi(
                                arith.CmpIPredicate.slt,
                                _raw(kv_start + key_offset),
                                _raw(valid_seq),
                            )
                            raw_scores.append(
                                ArithValue(key_in_bounds).select(
                                    score_f32, c_neg_inf
                                )
                            )

                local_max_raw = raw_scores[0]
                for item in range_constexpr(NUM_S_VALS - 1):
                    local_max_raw = _fmax(
                        local_max_raw, raw_scores[item + 1]
                    )
                row_max_raw = _fmax(
                    local_max_raw,
                    reduction_peer(local_max_raw),
                )
                row_max_offset = c_zero_f
                if const_expr(USE_FP8_P_OFFSET):
                    # Sage's 8.807 offset nearly fills E4M3 without crossing
                    # 448. The same factor enters P and L, so normalization
                    # cancels it.
                    row_max_offset = c_neg_fp8_p_offset
                row_max = fmath.fma(
                    row_max_raw,
                    _raw(score_scale),
                    _raw(row_max_offset),
                )
                m_new = _fmax(m_running, row_max)
                corr = rocdl.exp2(
                    T.f32,
                    _raw(_fsub(m_running, m_new)),
                )

                p_values = []
                neg_m_new = _fsub(c_zero_f, m_new)
                if const_expr(SOFTMAX_SUM_PARTS == 1):
                    # Preserve the production left-fold order exactly.
                    local_sum = _raw(c_zero_f)
                    for item in range_constexpr(NUM_S_VALS):
                        shifted = fmath.fma(
                            raw_scores[item],
                            _raw(score_scale),
                            neg_m_new,
                        )
                        probability = rocdl.exp2(T.f32, _raw(shifted))
                        p_values.append(probability)
                        local_sum = _fadd(local_sum, probability)
                else:
                    for item in range_constexpr(NUM_S_VALS):
                        shifted = fmath.fma(
                            raw_scores[item],
                            _raw(score_scale),
                            neg_m_new,
                        )
                        p_values.append(rocdl.exp2(T.f32, _raw(shifted)))
                    partial_sums = [
                        _raw(c_zero_f) for _ in range(SOFTMAX_SUM_PARTS)
                    ]
                    for item in range_constexpr(NUM_S_VALS):
                        part = item % SOFTMAX_SUM_PARTS
                        partial_sums[part] = _fadd(
                            partial_sums[part], p_values[item]
                        )
                    local_sum = partial_sums[0]
                    for part in range_constexpr(1, SOFTMAX_SUM_PARTS):
                        local_sum = _fadd(local_sum, partial_sums[part])

                tile_sum = _fadd(local_sum, reduction_peer(local_sum))
                l_new = _fadd(_fmul(corr, l_running), tile_sum)
                corr_vec = (
                    Vec.from_elements([corr], fx.Float32)
                    .broadcast_to(8)
                    .ir_value()
                )
                if const_expr(GATED_O_RESCALE):
                    # Keep the branch-carried values scalar in Python so
                    # FlyDSL forms vector SSA phis instead of 64 selects.
                    o_acc0 = o_accs[0]
                    o_acc1 = o_accs[1]
                    o_acc2 = o_accs[2]
                    o_acc3 = o_accs[3]
                    o_acc4 = o_accs[4]
                    o_acc5 = o_accs[5]
                    o_acc6 = o_accs[6]
                    o_acc7 = o_accs[7]
                    corr_ne_one = arith.cmpf(
                        arith.CmpFPredicate.ONE,
                        _raw(corr),
                        _raw(c_one_f),
                    )
                    if corr_ne_one:
                        o_acc0 = _fmul(o_acc0, corr_vec)
                        o_acc1 = _fmul(o_acc1, corr_vec)
                        o_acc2 = _fmul(o_acc2, corr_vec)
                        o_acc3 = _fmul(o_acc3, corr_vec)
                        o_acc4 = _fmul(o_acc4, corr_vec)
                        o_acc5 = _fmul(o_acc5, corr_vec)
                        o_acc6 = _fmul(o_acc6, corr_vec)
                        o_acc7 = _fmul(o_acc7, corr_vec)
                    o_accs = [
                        o_acc0,
                        o_acc1,
                        o_acc2,
                        o_acc3,
                        o_acc4,
                        o_acc5,
                        o_acc6,
                        o_acc7,
                    ]
                else:
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = _fmul(o_accs[dc], corr_vec)

                p_fragments = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    p_subtile = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = (st_idx * 2 + pks) * 8
                        p_subtile.append(
                            _pack_fp8_probability(
                                [
                                    p_values[p_base + item]
                                    for item in range(8)
                                ]
                            )
                        )
                    p_fragments.append(p_subtile)
                p_fragments_by_row.append(p_fragments)
                row_next_states.append([m_new, l_new, o_accs])

            if const_expr(
                not DB_KV_PIPELINE
                and prefetch_next
                and (PREFETCH_K or PREFETCH_V)
                and PREFETCH_STAGE == "before_pv"
            ):
                next_kv_start = kv_start + BLOCK_N
                safe_next_kv_start = _safe_tile_start(next_kv_start)
                if const_expr(PREFETCH_K):
                    next_k_prefetch = _load_k_tile_global(safe_next_kv_start)
                if const_expr(PREFETCH_V):
                    next_v_prefetch = _load_v_tile_global(safe_next_kv_start)

            if const_expr(EXPERIMENTAL_BARRIER_OVERLAP):
                # Load every V fragment before signaling so a faster wave can
                # never release peers to overwrite LDS while a slower wave is
                # still reading the current tile. A small prefix of PV work
                # overlaps the LDS drain; the suffix is register-only work
                # between the split barrier halves. R32 reuses each loaded V
                # fragment for both owned row groups.
                v_fragments = []
                for pks in range_constexpr(PV_K_STEPS):
                    pks_fragments = []
                    for dc in range_constexpr(D_CHUNKS):
                        dc_fragments = []
                        for st_idx in range_constexpr(N_SUB_TILES):
                            dc_fragments.append(
                                _load_v_fragment(
                                    st_idx,
                                    pks,
                                    dc,
                                    current_buffer,
                                )
                            )
                        pks_fragments.append(dc_fragments)
                    v_fragments.append(pks_fragments)

                pv_wmmas_total = (
                    PV_K_STEPS
                    * D_CHUNKS
                    * N_SUB_TILES
                    * ROW_GROUPS_PER_WAVE
                )
                # R16 intentionally reproduces the reserve-generated 13-PV
                # signal window. R32 retains the symmetric 16/16 split.
                pv_wmmas_in_window = (
                    13 if ROW_GROUPS_PER_WAVE == 1 else 16
                )
                pv_wmmas_before_signal = (
                    pv_wmmas_total - pv_wmmas_in_window
                )

                def _compute_pv_range(begin, end):
                    for pks in range_constexpr(PV_K_STEPS):
                        for dc in range_constexpr(D_CHUNKS):
                            for st_idx in range_constexpr(N_SUB_TILES):
                                packed_v = _pack_i8_fragment(
                                    v_fragments[pks][dc][st_idx]
                                )
                                for row_group in range_constexpr(
                                    ROW_GROUPS_PER_WAVE
                                ):
                                    ordinal = (
                                        (
                                            pks * D_CHUNKS * N_SUB_TILES
                                            + dc * N_SUB_TILES
                                            + st_idx
                                        )
                                        * ROW_GROUPS_PER_WAVE
                                        + row_group
                                    )
                                    if const_expr(begin <= ordinal < end):
                                        o_accs = row_next_states[row_group][2]
                                        p_fragments = p_fragments_by_row[
                                            row_group
                                        ]
                                        o_accs[dc] = _wmma_pv(
                                            packed_v,
                                            p_fragments[st_idx][pks],
                                            o_accs[dc],
                                        )

                _compute_pv_range(0, pv_wmmas_before_signal)

                _schedule_barrier()
                rocdl.s_wait_dscnt(0)
                rocdl.s_barrier_signal(-1)
                _schedule_barrier()

                _compute_pv_range(pv_wmmas_before_signal, pv_wmmas_total)

                _schedule_barrier()
                rocdl.s_barrier_wait(-1)
                _schedule_barrier()
            elif const_expr(PRE_LOAD_V):
                o_accs = row_next_states[0][2]
                p_fragments = p_fragments_by_row[0]
                current_v = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    current_v.append(
                        _load_v_fragment(st_idx, 0, 0, current_buffer)
                    )

                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        next_dc = dc + 1
                        next_pks = pks
                        if const_expr(next_dc >= D_CHUNKS):
                            next_dc = 0
                            next_pks = pks + 1
                        has_next = const_expr(next_pks < PV_K_STEPS)

                        next_v = []
                        if const_expr(has_next):
                            for st_idx in range_constexpr(N_SUB_TILES):
                                next_v.append(
                                    _load_v_fragment(
                                        st_idx,
                                        next_pks,
                                        next_dc,
                                        current_buffer,
                                    )
                                )

                        for st_idx in range_constexpr(N_SUB_TILES):
                            o_accs[dc] = _wmma_pv(
                                _pack_i8_fragment(current_v[st_idx]),
                                p_fragments[st_idx][pks],
                                o_accs[dc],
                            )

                        if const_expr(has_next):
                            current_v = next_v
            else:
                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        for st_idx in range_constexpr(N_SUB_TILES):
                            v_bytes = _load_v_fragment(
                                st_idx,
                                pks,
                                dc,
                                current_buffer,
                            )
                            packed_v = _pack_i8_fragment(v_bytes)
                            for row_group in range_constexpr(
                                ROW_GROUPS_PER_WAVE
                            ):
                                o_accs = row_next_states[row_group][2]
                                p_fragments = p_fragments_by_row[row_group]
                                o_accs[dc] = _wmma_pv(
                                    packed_v,
                                    p_fragments[st_idx][pks],
                                    o_accs[dc],
                                )

            if const_expr(
                not DB_KV_PIPELINE and not EXPERIMENTAL_BARRIER_OVERLAP
            ):
                gpu.barrier()
            next_args = []
            for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
                m_new, l_new, o_accs = row_next_states[row_group]
                next_args.extend([m_new, l_new])
                next_args.extend(o_accs)
            if const_expr(S_LOOKAHEAD):
                for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
                    for s_acc in range_constexpr(NUM_S_ACCS):
                        next_args.append(next_s_lookahead[row_group][s_acc])
            if const_expr(carry_prefetch and PREFETCH_K):
                for load_batch in range_constexpr(K_LOAD_BATCHES):
                    next_args.append(next_k_prefetch[load_batch])
            if const_expr(carry_prefetch and PREFETCH_V):
                for load_batch in range_constexpr(V_LOAD_BATCHES):
                    next_args.append(next_v_prefetch[load_batch])
            return next_args

        loop_results = init_args
        if const_expr(KEY_TAIL_PEEL_SEQ_LEN > 0):
            peel_tail_start = fx.Index(
                KEY_TAIL_PEEL_SEQ_LEN
                - KEY_TAIL_PEEL_SEQ_LEN % BLOCK_N
            )
            for kv_start, inner in range(
                0,
                peel_tail_start,
                BLOCK_N,
                init=init_args,
            ):
                next_args = _process_kv_tile(
                    kv_start,
                    inner,
                    mask_key_tail=False,
                    prefetch_next=True,
                    carry_prefetch=True,
                )
                loop_results = yield next_args
            loop_results = _process_kv_tile(
                peel_tail_start,
                loop_results,
                mask_key_tail=True,
                prefetch_next=False,
                carry_prefetch=False,
            )
        else:
            for kv_start, inner in range(
                0,
                seq,
                BLOCK_N,
                init=init_args,
            ):
                next_args = _process_kv_tile(
                    kv_start,
                    inner,
                    mask_key_tail=not NO_KEY_TAIL,
                    prefetch_next=True,
                    carry_prefetch=True,
                )
                loop_results = yield next_args

        for row_group in range_constexpr(ROW_GROUPS_PER_WAVE):
            state_base = row_group * ROW_STATE_WIDTH
            inv_l = arith.divf(
                _raw(c_one_f),
                _raw(loop_results[state_base + 1]),
                fastmath=fm_fast,
            )
            inv_l_vec = (
                Vec.from_elements([inv_l], fx.Float32)
                .broadcast_to(8)
                .ir_value()
            )
            if q_in_bounds[row_group]:
                if const_expr(RETURN_LSE):
                    if klane == 0:
                        log2_l = rocdl.log(
                            T.f32, _raw(loop_results[state_base + 1])
                        )
                        lse = _fmul(
                            _fadd(loop_results[state_base], log2_l), c_ln2
                        )
                        _global_store(
                            lse_ptr,
                            lse_global_index(q_rows[row_group]),
                            T.f32,
                            lse,
                        )
                for dc in range_constexpr(D_CHUNKS):
                    d_col = fx.Index(dc * GFX1201_WMMA_N) + klane * 8
                    scale_index = (
                        (batch_idx * num_heads + head_idx) * head_dim + d_col
                    )
                    v_scale = _global_load(
                        vs_ptr,
                        scale_index,
                        T.f32,
                        v8f32_type,
                    )
                    normalized = _fmul(
                        loop_results[state_base + 2 + dc], inv_l_vec
                    )
                    scaled = _fmul(normalized, v_scale)
                    out = Vec(scaled).to(output_numeric).ir_value()
                    _global_store(
                        o_ptr,
                        qk_global_index(q_rows[row_group], d_col),
                        output_numeric.ir_type,
                        out,
                    )

    @flyc.jit
    def launch_sage_attention_core(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        QScale: fx.Pointer,
        KScale: fx.Pointer,
        VScale: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        batch_size: fx.Int32,
        padded_seq_len: fx.Int32,
        valid_seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        valid_seq = fx.Index(valid_seq_len)
        grid_x = (
            fx.Index(batch_size)
            * ((valid_seq + BLOCK_M - 1) // BLOCK_M)
            * num_heads
        )
        if const_expr(EXPERIMENTAL_GRID_BLOCKS > 0):
            grid_x = fx.Index(EXPERIMENTAL_GRID_BLOCKS)
        launcher = sage_attention_core(
            Q,
            K,
            V,
            QScale,
            KScale,
            VScale,
            O,
            LSE,
            padded_seq_len,
            valid_seq_len,
        )
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch_sage_attention_core.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _ptr_arg(value):
        if value is None:
            return flyc.from_c_void_p(fx.Uint8, 0)
        if hasattr(value, "data_ptr"):
            type_name = type(value).__name__
            module_name = type(value).__module__
            pointer = (
                0
                if type_name == "FakeTensor" or "fake_tensor" in module_name
                else value.data_ptr()
            )
            return flyc.from_c_void_p(fx.Uint8, pointer)
        return value

    def _launch(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        out,
        batch_size,
        padded_seq_len,
        valid_seq_len,
        stream=None,
        lse=None,
    ):
        args = tuple(
            _ptr_arg(value)
            for value in (q, k, v, q_scale, k_scale, v_scale, out, lse)
        )
        compiled = getattr(launch_sage_attention_core, "_compiled", None)
        runtime_args = (
            *args,
            batch_size,
            padded_seq_len,
            valid_seq_len,
            fx.Stream(stream),
        )
        if compiled is None:
            compiled = flyc.compile(launch_sage_attention_core, *runtime_args)
            launch_sage_attention_core._compiled = compiled
        else:
            compiled(*runtime_args)

    _launch.jit_function = launch_sage_attention_core
    return _launch


def compile_sage_wmma_probes():
    """Compile both probes using null pointers; intended for COMPILE_ONLY=1."""

    null_i8 = flyc.from_c_void_p(fx.Uint8, 0)
    stream = fx.Stream(None)
    build_sage_wmma_qk_probe()(null_i8, null_i8, null_i8, stream)
    build_sage_wmma_pv_probe()(null_i8, null_i8, null_i8, stream)


def verify_dumped_wmma_isa(dump_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """Find and validate the final ISA files emitted for both probes."""

    root = Path(dump_dir)
    expected = {
        _QK_PROBE_NAME: _QK_ISA,
        _PV_PROBE_NAME: _PV_ISA,
    }
    found: dict[str, Path] = {}
    for kernel_name, instruction in expected.items():
        candidates = sorted((root / kernel_name).glob("*_final_isa.s"))
        if not candidates:
            candidates = sorted((root / kernel_name).glob("*.s"))
        if not candidates:
            raise RuntimeError(f"no ISA dump found for {kernel_name} under {root}")
        isa_path = candidates[-1]
        isa = isa_path.read_text(encoding="utf-8").lower()
        if instruction not in isa:
            raise RuntimeError(
                f"{kernel_name} did not lower to {instruction}; inspect {isa_path}"
            )
        found[kernel_name] = isa_path
    return found


def _main():
    parser = argparse.ArgumentParser(description="Compile gfx1201 SageAttention2 WMMA probes")
    parser.add_argument("--dump-dir", default="/tmp/sage_gfx1201_wmma_isa")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("COMPILE_ONLY", "1")
    os.environ.setdefault("ARCH", "gfx1201")
    os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1201")
    os.environ.setdefault("FLYDSL_DUMP_IR", "1")
    os.environ.setdefault("FLYDSL_DUMP_DIR", args.dump_dir)

    compile_sage_wmma_probes()
    if args.verify:
        for name, path in verify_dumped_wmma_isa(args.dump_dir).items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    _main()
