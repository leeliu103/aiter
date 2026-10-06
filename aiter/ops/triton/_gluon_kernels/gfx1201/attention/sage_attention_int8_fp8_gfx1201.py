# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Noncausal INT8/FP8 SageAttention for gfx1201.

All tensors are contiguous, with the following prepared layouts:
    Q, K:          INT8   [B, P, H, 128]
    V:             E4M3FN [B, H, 128, P]
    QScale, KScale: FP32   [B, H, P/32]
    VScale:        FP32   [B, H, 128]
    Out:           BF16 or FP16 [B, P, H, 128]
    LSE (optional): FP32  [B, H, P]

P = ceil(SEQ_LEN / 32) * 32 and H = NUM_HEADS. QScale includes
softmax_scale * log2(e). Valid Q/K scales must be positive and finite.
K/V padding is read unmasked and must contain finite, initialized values.
Only valid query rows are written. Q/K addressing requires B*P*H*128 < 2**31.

Launch B * H * ceil(SEQ_LEN / 128) programs with LAUNCH_OPTIONS. Consecutive
programs handle consecutive 128-query tiles within a head; each program
streams all keys in 32-key tiles. LSE is for the supplied (centered) keys.

The numerical ordering follows the M128/N32 FlyDSL kernel in leeliu103/aiter
(sage-attention, adf0def), including its lane-specific softmax sums and FP8
rounding.
Requires Triton with Gluon RDNA4 WMMA support.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.cdna3 import buffer_load
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
    # Sum in register order; a tree reduction changes FP32 rounding.
    values = (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
    total = values[0] + values[1]
    for i in gl.static_range(2, 16):
        total = total + values[i]
    # map_elementwise keeps the input shape. The caller selects one copy.
    return (total,) * 16


@gluon.jit
def _expand_row_pairs(row_values):
    """Broadcast each row's two lane values to their owning WMMA columns."""
    # Restore column ownership: lane 0..15 owns 0..7 and 16..23;
    # lane 16..31 owns 8..15 and 24..31.
    expanded = gl.reshape(row_values, [128, 1, 2, 1])
    expanded = expanded.broadcast_to((128, 2, 2, 8))
    return gl.convert_layout(
        gl.reshape(expanded, [128, 32]), _WMMA_LAYOUT, assert_trivial=True
    )


# One pack covers all 64 accumulator values owned by a lane.
# The 64 scale arguments are copies of that row's scale; only scale0 is needed.
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
    scale0, *unused_scales,
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
    K, key_start, batch_head, PADDED_LEN: gl.constexpr, NUM_HEADS: gl.constexpr
):
    rows = key_start + gl.arange(0, 32, layout=gl.SliceLayout(1, _K_LOAD_LAYOUT))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, _K_LOAD_LAYOUT))
    # A scalar head base lets buffer loads use 32-bit per-lane offsets.
    base = (
        K
        + (batch_head // NUM_HEADS * PADDED_LEN * NUM_HEADS + batch_head % NUM_HEADS)
        * 128
    )
    offsets = rows[:, None] * (NUM_HEADS * 128) + cols[None, :]
    return buffer_load(base, offsets)


@gluon.jit
def _load_v_tile(V, key_start, batch_head, PADDED_LEN: gl.constexpr):
    rows = key_start + gl.arange(0, 32, layout=gl.SliceLayout(1, _V_LOAD_LAYOUT))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, _V_LOAD_LAYOUT))
    base = V + batch_head * 128 * PADDED_LEN
    offsets = cols[None, :] * PADDED_LEN + rows[:, None]
    return buffer_load(base, offsets)


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
    shifted_max,
    denominator,
    key_start,
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
        next_key_start = key_start + 32
        next_k = _load_k_tile(K, next_key_start, batch_head, PADDED_LEN, NUM_HEADS)
        next_v = _load_v_tile(V, next_key_start, batch_head, PADDED_LEN)
    else:
        next_k = loaded_k
        next_v = loaded_v

    k_fragment = k_shared.permute((1, 0)).load(gl.DotOperandLayout(1, _WMMA_LAYOUT, 8))
    v_fragment = v_shared.load(gl.DotOperandLayout(1, _WMMA_LAYOUT, 8))
    scores = wmma(
        q_fragment, k_fragment, gl.full([128, 32], 0, gl.int32, _WMMA_LAYOUT)
    ).to(gl.float32)
    cols = gl.arange(0, 32, layout=gl.SliceLayout(0, _WMMA_LAYOUT))
    if MASK_TAIL:
        scores = gl.where(key_start + cols[None, :] < SEQ_LEN, scores, float("-inf"))
    k_scale = gl.load(KScale + batch_head * (PADDED_LEN // 32) + key_start // 32)
    scale = q_scale * k_scale

    # Track the largest base-2 logit minus 8.807, so the largest unnormalized
    # weight is near FP8's maximum (448). Preserve the literal and FMA order:
    # replacing 8.807 with log2(448) changes the reference's rounding.
    tile_shifted_max = gl.fma(gl.max(scores, 1), scale, -8.807)
    new_shifted_max = gl.maximum(shifted_max, tile_shifted_max)
    alpha = gl.exp2(shifted_max - new_shifted_max)
    weights = gl.exp2(gl.fma(scores, scale[:, None], -new_shifted_max[:, None]))
    # Each row spans two lanes, owning keys 0..7,16..23 and 8..15,24..31.
    local_sum = gl.map_elementwise(_sum16, weights, pack=16)[0]
    # Collapse register copies to one scalar per lane. These maxima only
    # select identical copies; they perform no additional probability sum.
    local_sum = gl.max(gl.max(gl.reshape(local_sum, [128, 2, 2, 8]), 3), 1)
    local_sum = gl.convert_layout(local_sum, _ROW_PAIR_LAYOUT, assert_trivial=True)
    # Exchange corresponding lanes in the wave's two 16-lane halves.
    peer_sum = gl.inline_asm_elementwise(
        "v_permlanex16_b32 $0, $1, $2, 0xfedcba98 op_sel:[1,0]",
        constraints="=v,v,s",
        args=[local_sum, 0x76543210],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    # The two lanes add local/peer sums in opposite orders. Keep both denominators
    # and this FMA/add sequence to preserve their distinct FP32 rounding.
    row_rescale = gl.convert_layout(
        alpha, gl.SliceLayout(1, _ROW_PAIR_LAYOUT), assert_trivial=True
    )
    denominator = gl.fma(row_rescale[:, None], denominator, peer_sum) + local_sum
    # The denominator uses FP32 weights; only the PV operand is rounded to FP8.
    fp8_weights = gl.map_elementwise(_convert_four_fp8, weights, pack=4)[0].to(
        gl.float8e4nv, bitcast=True
    )
    acc = gl.map_elementwise(_rescale_accumulator64, acc, alpha[:, None], pack=64)[0]
    p_fragment = gl.convert_layout(fp8_weights, gl.DotOperandLayout(0, _WMMA_LAYOUT, 8))
    acc = wmma(p_fragment, v_fragment, acc)
    # Hold both allocations through the barrier: every wave must finish its
    # reads before the next iteration can reuse this LDS storage.
    gl.barrier()
    k_shared._keep_alive()
    v_shared._keep_alive()
    return acc, new_shifted_max, denominator, next_k, next_v


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
    gl.static_assert(SEQ_LEN > 0)
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
    shifted_max = gl.full(
        [128], float("-inf"), gl.float32, gl.SliceLayout(1, _WMMA_LAYOUT)
    )
    denominator = gl.full([128, 2], 0.0, gl.float32, _ROW_PAIR_LAYOUT)
    acc = gl.full([128, 128], 0.0, gl.float32, _WMMA_LAYOUT)
    loaded_k = _load_k_tile(K, 0, batch_head, PADDED_LEN, NUM_HEADS)
    loaded_v = _load_v_tile(V, 0, batch_head, PADDED_LEN)

    # Every loop iteration has a successor; the last tile needs no prefetch.
    last_key_start: gl.constexpr = PADDED_LEN - 32
    for key_start in range(0, last_key_start, 32):
        acc, shifted_max, denominator, loaded_k, loaded_v = _attention_tile(
            q_fragment,
            K,
            V,
            KScale,
            q_scale,
            loaded_k,
            loaded_v,
            acc,
            shifted_max,
            denominator,
            key_start,
            batch_head,
            SEQ_LEN,
            PADDED_LEN,
            NUM_HEADS,
            MASK_TAIL=False,
            PREFETCH=True,
        )
    acc, shifted_max, denominator, _, _ = _attention_tile(
        q_fragment,
        K,
        V,
        KScale,
        q_scale,
        loaded_k,
        loaded_v,
        acc,
        shifted_max,
        denominator,
        last_key_start,
        batch_head,
        SEQ_LEN,
        PADDED_LEN,
        NUM_HEADS,
        MASK_TAIL=SEQ_LEN % 32 != 0,
        PREFETCH=False,
    )

    # Match the reference's native reciprocal rather than a division sequence.
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
    cols = gl.arange(0, 32, layout=gl.SliceLayout(0, _WMMA_LAYOUT))
    for part in gl.static_range(4):
        partial = (o0, o1, o2, o3)[part]
        channels = part * 32 + cols
        v_scale = gl.load(VScale + batch_head * 128 + channels)
        # Preserve the two rounding steps: normalize first, then apply VScale.
        out = (partial * inv_denominator) * v_scale[None, :]
        offsets = (
            (batch_head // NUM_HEADS * PADDED_LEN + rows[:, None]) * NUM_HEADS
            + batch_head % NUM_HEADS
        ) * 128 + channels[None, :]
        gl.store(Out + offsets, out, rows[:, None] < SEQ_LEN)
    if RETURN_LSE:
        # The reference takes LSE from the first lane's denominator.
        denominator = _expand_row_pairs(denominator)
        lse_denominator = gl.sum(gl.where(cols[None, :] == 0, denominator, 0.0), 1)
        lse = (shifted_max + gl.log2(lse_denominator)) * 0.6931471805599453
        gl.store(LSE + batch_head * PADDED_LEN + rows, lse, rows < SEQ_LEN)
