# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Head-major INT8/FP8 SageAttention for gfx1201, implemented in Gluon.

Ports the M128/N32 path at leeliu103/aiter sage-attention commit adf0def.
Q/K: contiguous INT8 [B, P, H, 128]. V: E4M3FN [B, H, 128, P].
QScale/KScale: FP32 [B, H, P/32]; QScale includes softmax_scale * log2(e).
VScale: FP32 [B, H, 128]. Out: BF16/FP16 [B, P, H, 128].
Optional LSE: FP32 [B, H, P], for the supplied (centered) keys.
Here P is PADDED_LEN and H is NUM_HEADS.

Each program keeps 128 queries in registers and streams 32-key K/V tiles.
An iteration computes INT8 QK, updates online softmax, then accumulates FP8 PV.
The helpers preserve the reference's rounding order and register ownership.

Only valid query rows are written. All input padding must be initialized.
The implementation needs Triton with Gluon RDNA4 WMMA support.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna4 import wmma

BLOCK_M = 128
LAUNCH_OPTIONS = {"num_warps": 8, "num_stages": 1, "waves_per_eu": 0}

# Explicit constexpr wrappers include layouts in Triton's compilation key.
# Eight waves span the query rows; each lane owns 64 accumulator channels.
_WMMA_LAYOUT = gl.constexpr(
    AMDWMMALayout(2, True, [[1, 0], [2, 0], [4, 0]], instr_shape=[16, 16, 16])
)
# A row has two lane-specific denominators. Their FP32 rounding can differ.
_ROW_PAIR_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 1], [16, 2], [8, 1], [0, 1]))
_K_LOAD_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 16], [4, 8], [8, 1], [1, 0]))
_V_LOAD_LAYOUT = gl.constexpr(gl.BlockedLayout([16, 1], [2, 16], [1, 8], [0, 1]))
_K_SHARED_LAYOUT = gl.constexpr(gl.SwizzledSharedLayout(8, 1, 16, [1, 0]))
# Fold the V tile to enable 128-bit LDS loads without padding. Together with K,
# this uses 8 KiB of shared memory while preserving the WMMA operand ordering.
# Each basis maps one shared-memory address bit to [key row, channel].
_V_SHARED_LAYOUT = gl.constexpr(
    gl.SharedLinearLayout(
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [0, 64],
            [16, 0],
            [0, 1],
            [0, 2],
            [16, 4],
            [8, 0],
            [0, 8],
            [0, 16],
            [0, 32],
        ],
        alignment=16,
    )
)


@gluon.jit
def _convert_four_fp8(x0, x1, x2, x3):
    # Native conversion avoids this Triton revision's software E4M3 cast.
    word = gl.inline_asm_elementwise(
        "v_cvt_pk_fp8_f32 $0, $1, $2\n" "v_cvt_pk_fp8_f32 $0, $3, $4 op_sel:[0,0,1]",
        constraints="=&v,v,v,v,v",
        args=[x0, x1, x2, x3],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )
    return (
        word.to(gl.uint8),
        (word >> 8).to(gl.uint8),
        (word >> 16).to(gl.uint8),
        (word >> 24).to(gl.uint8),
    )


@gluon.jit
def _probabilities_to_fp8(x):
    return gl.map_elementwise(_convert_four_fp8, x, pack=4)[0].to(
        gl.float8e4nv, bitcast=True
    )


@gluon.jit
def _split_channels(x):
    """Split the channel axis in half without moving values between lanes."""
    halves = gl.permute(gl.reshape(x, [128, 2, x.shape[1] // 2]), (0, 2, 1))
    left, right = gl.split(halves)
    return (
        gl.convert_layout(left, _WMMA_LAYOUT, assert_trivial=True),
        gl.convert_layout(right, _WMMA_LAYOUT, assert_trivial=True),
    )


@gluon.jit
def _sum16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    # Preserve FlyDSL's left fold. A tree reduction changes output rounding.
    values = (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
    total = values[0] + values[1]
    for i in gl.static_range(2, 16):
        total = total + values[i]
    return (total,) * 16


@gluon.jit
def _lane_probability_sums(probabilities):
    # The two lanes own keys 0..7,16..23 and 8..15,24..31, respectively.
    local = gl.map_elementwise(_sum16, probabilities, pack=16)[0]
    # Collapse register copies to one scalar per lane. These maxima only
    # select identical copies; they perform no additional probability sum.
    local = gl.max(gl.max(gl.reshape(local, [128, 2, 2, 8]), 3), 1)
    local = gl.convert_layout(local, _ROW_PAIR_LAYOUT, assert_trivial=True)
    peer = gl.inline_asm_elementwise(
        "v_permlanex16_b32 $0, $1, $2, 0xfedcba98 op_sel:[1,0]",
        constraints="=v,v,s",
        args=[local, 0x76543210],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return local, peer


@gluon.jit
def _expand_row_pairs(denominator):
    """Expand [128, 2] lane values into the [128, 32] WMMA column layout."""
    # Restore column ownership: lane 0..15 owns 0..7 and 16..23;
    # lane 16..31 owns 8..15 and 24..31.
    expanded = gl.reshape(denominator, [128, 1, 2, 1])
    expanded, _ = gl.broadcast(
        expanded, gl.full([128, 2, 2, 8], 0.0, gl.float32, expanded.type.layout)
    )
    return gl.convert_layout(
        gl.reshape(expanded, [128, 32]), _WMMA_LAYOUT, assert_trivial=True
    )


# A pack contains one lane's 64 accumulator values, followed by 64 copies of
# its row's rescale factor. One branch skips all multiplies when scale == 1.
# Keep this pack size: scalar mapping emits slower code on this Triton build.
# fmt: off
@gluon.jit
def _rescale_accumulator64(
    a0, a1, a2, a3, a4, a5, a6, a7,
    a8, a9, a10, a11, a12, a13, a14, a15,
    a16, a17, a18, a19, a20, a21, a22, a23,
    a24, a25, a26, a27, a28, a29, a30, a31,
    a32, a33, a34, a35, a36, a37, a38, a39,
    a40, a41, a42, a43, a44, a45, a46, a47,
    a48, a49, a50, a51, a52, a53, a54, a55,
    a56, a57, a58, a59, a60, a61, a62, a63,
    scale0, scale1, scale2, scale3, scale4, scale5, scale6, scale7,
    scale8, scale9, scale10, scale11, scale12, scale13, scale14, scale15,
    scale16, scale17, scale18, scale19, scale20, scale21, scale22, scale23,
    scale24, scale25, scale26, scale27, scale28, scale29, scale30, scale31,
    scale32, scale33, scale34, scale35, scale36, scale37, scale38, scale39,
    scale40, scale41, scale42, scale43, scale44, scale45, scale46, scale47,
    scale48, scale49, scale50, scale51, scale52, scale53, scale54, scale55,
    scale56, scale57, scale58, scale59, scale60, scale61, scale62, scale63,
):
    values = (
        a0, a1, a2, a3, a4, a5, a6, a7,
        a8, a9, a10, a11, a12, a13, a14, a15,
        a16, a17, a18, a19, a20, a21, a22, a23,
        a24, a25, a26, a27, a28, a29, a30, a31,
        a32, a33, a34, a35, a36, a37, a38, a39,
        a40, a41, a42, a43, a44, a45, a46, a47,
        a48, a49, a50, a51, a52, a53, a54, a55,
        a56, a57, a58, a59, a60, a61, a62, a63,
    )
    if scale0 != 1.0:
        # Gluon unrolls this tuple comprehension at compile time.
        values = [value * scale0 for value in values]
    return values
# fmt: on


@gluon.jit
def _load_k_tile(
    K, start, batch_head, PADDED_LEN: gl.constexpr, NUM_HEADS: gl.constexpr
):
    rows = start + gl.arange(0, 32, layout=gl.SliceLayout(1, _K_LOAD_LAYOUT))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, _K_LOAD_LAYOUT))
    offsets = (
        (batch_head // NUM_HEADS * PADDED_LEN + rows[:, None]) * NUM_HEADS
        + batch_head % NUM_HEADS
    ) * 128 + cols[None, :]
    return gl.load(K + offsets)


@gluon.jit
def _load_v_tile(V, start, batch_head, PADDED_LEN: gl.constexpr):
    rows = start + gl.arange(0, 32, layout=gl.SliceLayout(1, _V_LOAD_LAYOUT))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, _V_LOAD_LAYOUT))
    return gl.load(V + (batch_head * 128 + cols[None, :]) * PADDED_LEN + rows[:, None])


@gluon.jit
def _attention_tile(
    q_fragment,
    K,
    V,
    KScale,
    q_scale,
    loaded_k,
    loaded_v,
    acc,
    row_max,
    denominator,
    start,
    batch_head,
    SEQ_LEN: gl.constexpr,
    PADDED_LEN: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    MASK_TAIL: gl.constexpr,
    PREFETCH: gl.constexpr,
):
    """Consume one K/V tile and return updated attention state and prefetched K/V."""
    # Store both tiles before reading K, and prefetch the next pair during QK.
    k_shared = gl.allocate_shared_memory(gl.int8, [32, 128], _K_SHARED_LAYOUT, loaded_k)
    v_shared = gl.allocate_shared_memory(
        gl.float8e4nv, [32, 128], _V_SHARED_LAYOUT, loaded_v
    )
    # Keep prefetches after the producer barrier so its global-memory
    # fence does not wait for the next tile's loads.
    gl.barrier()
    if PREFETCH:
        next_start = gl.where(start + 32 < PADDED_LEN, start + 32, 0)
        next_k = _load_k_tile(K, next_start, batch_head, PADDED_LEN, NUM_HEADS)
        next_v = _load_v_tile(V, next_start, batch_head, PADDED_LEN)
    else:
        next_k = loaded_k
        next_v = loaded_v

    k_fragment = k_shared.permute((1, 0)).load(gl.DotOperandLayout(1, _WMMA_LAYOUT, 8))
    scores = wmma(
        q_fragment, k_fragment, gl.full([128, 32], 0, gl.int32, _WMMA_LAYOUT)
    ).to(gl.float32)
    cols = gl.arange(0, 32, layout=gl.SliceLayout(0, _WMMA_LAYOUT))
    if MASK_TAIL:
        scores = gl.where(start + cols[None, :] < SEQ_LEN, scores, float("-inf"))
    k_scale = gl.load(KScale + batch_head * (PADDED_LEN // 32) + start // 32)
    scale = q_scale * k_scale

    # The offset puts the largest probability near FP8's maximum value (448).
    # Keep it inside FMA, as in the reference.
    # The denominator uses the unrounded probabilities.
    tile_max = gl.fma(gl.max(scores, 1), scale, -8.807)
    new_max = gl.maximum(row_max, tile_max)
    alpha = gl.exp2(row_max - new_max)
    probabilities = gl.exp2(gl.fma(scores, scale[:, None], -new_max[:, None]))
    local_sum, peer_sum = _lane_probability_sums(probabilities)
    # FlyDSL's fast-math backend forms (alpha * denominator + peer_sum) + local_sum.
    # Keep both lane denominators: rounding can make their values different.
    row_rescale = gl.convert_layout(
        alpha, gl.SliceLayout(1, _ROW_PAIR_LAYOUT), assert_trivial=True
    )
    denominator = gl.fma(row_rescale[:, None], denominator, peer_sum) + local_sum
    fp8_probabilities = _probabilities_to_fp8(probabilities)
    acc = gl.map_elementwise(_rescale_accumulator64, acc, alpha[:, None], pack=64)[0]
    v_fragment = v_shared.load(gl.DotOperandLayout(1, _WMMA_LAYOUT, 8))
    p_fragment = gl.convert_layout(
        fp8_probabilities, gl.DotOperandLayout(0, _WMMA_LAYOUT, 8)
    )
    acc = wmma(p_fragment, v_fragment, acc)
    # Finish both shared-memory reads before the next iteration reuses LDS.
    gl.barrier()
    k_shared._keep_alive()
    v_shared._keep_alive()
    return acc, new_max, denominator, next_k, next_v


@gluon.jit
def _store_output(
    acc,
    row_max,
    denominator,
    VScale,
    Out,
    LSE,
    rows,
    batch_head,
    SEQ_LEN: gl.constexpr,
    PADDED_LEN: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    RETURN_LSE: gl.constexpr,
):
    """Normalize four channel groups, then optionally store natural-log LSE."""
    inv_denominator = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1;",
        constraints="=v,v",
        args=[denominator],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    inv_denominator = _expand_row_pairs(inv_denominator)
    left, right = _split_channels(acc)
    o0, o1 = _split_channels(left)
    o2, o3 = _split_channels(right)
    for part in gl.static_range(4):
        partial = (o0, o1, o2, o3)[part]
        channels = part * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, _WMMA_LAYOUT))
        v_scale = gl.load(VScale + batch_head * 128 + channels)
        out = (partial * inv_denominator) * v_scale[None, :]
        offsets = (
            (batch_head // NUM_HEADS * PADDED_LEN + rows[:, None]) * NUM_HEADS
            + batch_head % NUM_HEADS
        ) * 128 + channels[None, :]
        gl.store(Out + offsets, out, rows[:, None] < SEQ_LEN)
    if RETURN_LSE:
        # The reference takes LSE from the first lane's denominator.
        denominator = _expand_row_pairs(denominator)
        col = gl.arange(0, 32, layout=gl.SliceLayout(0, _WMMA_LAYOUT))
        lse_denominator = gl.sum(gl.where(col[None, :] == 0, denominator, 0.0), 1)
        lse = (row_max + gl.log2(lse_denominator)) * 0.6931471805599453
        gl.store(LSE + batch_head * PADDED_LEN + rows, lse, rows < SEQ_LEN)


@gluon.jit
def _sage_attention_int8_fp8(
    Q,
    K,
    V,
    QScale,
    KScale,
    VScale,
    Out,
    LSE,
    SEQ_LEN: gl.constexpr,
    PADDED_LEN: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    RETURN_LSE: gl.constexpr = False,
):
    gl.static_assert(SEQ_LEN > 0 and PADDED_LEN >= SEQ_LEN and PADDED_LEN % 32 == 0)
    gl.static_assert(PADDED_LEN == ((SEQ_LEN + 31) // 32) * 32)
    gl.static_assert(Q.dtype.element_ty == gl.int8 and K.dtype.element_ty == gl.int8)
    gl.static_assert(V.dtype.element_ty == gl.float8e4nv)
    gl.static_assert(QScale.dtype.element_ty == gl.float32)
    gl.static_assert(KScale.dtype.element_ty == gl.float32)
    gl.static_assert(VScale.dtype.element_ty == gl.float32)
    if RETURN_LSE:
        gl.static_assert(LSE.dtype.element_ty == gl.float32)
    gl.static_assert(
        Out.dtype.element_ty == gl.bfloat16 or Out.dtype.element_ty == gl.float16
    )
    q_tiles: gl.constexpr = (SEQ_LEN + 127) // 128
    block = gl.program_id(0)
    query_tile = block % q_tiles
    batch_head = block // q_tiles
    # Consecutive programs process consecutive query tiles of the same head.
    rows = query_tile * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, _WMMA_LAYOUT))
    # Load Q directly into WMMA fragments, eight adjacent channels per lane.
    q_layout: gl.constexpr = gl.DotOperandLayout(0, _WMMA_LAYOUT, 8)
    q_rows = query_tile * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, q_layout))
    q_channels = gl.arange(0, 128, layout=gl.SliceLayout(0, q_layout))
    q_offsets = (
        (batch_head // NUM_HEADS * PADDED_LEN + q_rows[:, None]) * NUM_HEADS
        + batch_head % NUM_HEADS
    ) * 128 + q_channels[None, :]
    q_fragment = gl.load(Q + q_offsets, q_rows[:, None] < SEQ_LEN, 0)
    q_scale = gl.load(
        QScale + batch_head * (PADDED_LEN // 32) + rows // 32, rows < SEQ_LEN, 0.0
    )
    row_max = gl.full([128], float("-inf"), gl.float32, gl.SliceLayout(1, _WMMA_LAYOUT))
    denominator = gl.full([128, 2], 0.0, gl.float32, _ROW_PAIR_LAYOUT)
    acc = gl.full([128, 128], 0.0, gl.float32, _WMMA_LAYOUT)
    loaded_k = _load_k_tile(K, 0, batch_head, PADDED_LEN, NUM_HEADS)
    loaded_v = _load_v_tile(V, 0, batch_head, PADDED_LEN)

    # Full tiles avoid masks in the hot loop; only the final partial tile masks K.
    for start in range(0, SEQ_LEN // 32 * 32, 32):
        acc, row_max, denominator, loaded_k, loaded_v = _attention_tile(
            q_fragment,
            K,
            V,
            KScale,
            q_scale,
            loaded_k,
            loaded_v,
            acc,
            row_max,
            denominator,
            start,
            batch_head,
            SEQ_LEN,
            PADDED_LEN,
            NUM_HEADS,
            MASK_TAIL=False,
            PREFETCH=True,
        )
    if SEQ_LEN % 32:
        acc, row_max, denominator, loaded_k, loaded_v = _attention_tile(
            q_fragment,
            K,
            V,
            KScale,
            q_scale,
            loaded_k,
            loaded_v,
            acc,
            row_max,
            denominator,
            SEQ_LEN // 32 * 32,
            batch_head,
            SEQ_LEN,
            PADDED_LEN,
            NUM_HEADS,
            MASK_TAIL=True,
            PREFETCH=False,
        )

    _store_output(
        acc,
        row_max,
        denominator,
        VScale,
        Out,
        LSE,
        rows,
        batch_head,
        SEQ_LEN,
        PADDED_LEN,
        NUM_HEADS,
        RETURN_LSE,
    )
