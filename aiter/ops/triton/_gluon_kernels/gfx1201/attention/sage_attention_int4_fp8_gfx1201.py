"""Noncausal D128 SageAttention-style INT4 QK / FP8 PV on gfx1201.

Kernel only: the caller supplies quantized/centered tensors and corrections.
There is no quantizer, allocation, dispatch, or dependency on AITER helpers.
Tested with Triton 3.7.0+git5f3f125e; the INT4 builtin uses private compiler APIs
because this version has native i4 WMMA lowering but no Python i4 dtype.

The execution tile is always 128 queries by 32 keys. QB (128 or 256) is
the Q-centering block size, not the execution tile size. N is the logical
sequence length; NP = ceil(N / QB) * QB includes initialized padding.
All buffers are contiguous, with BH = B * H:

    Q, K       uint8 [BH, NP, 64], signed INT4 packed low/even nibble first
    V          E4M3FN [BH, 128, NP], centered, channel-major values
    QS, KS     float32 [BH, NP], expanded thread-group scales
    VS         float32 [BH, 128], per-channel V scales
    DELTA      float32 [BH, NP / QB, NP], Q-block-mean score correction
    MU, EV     float32 [BH, 128], original V mean and residual-error mean
    Q_ORIG     BF16/FP16 [B, N, H, 128], original Q, only needed for LSE
    KM         float32 [BH, 128], original K mean, only needed for LSE
    O          BF16/FP16 [B, N, H, 128], output
    LSE        float32 [B, H, N], optional natural-log normalization

Q/K are centered before symmetric INT4 quantization to [-7, 7]. Q scales
share rows r and r+64 in each M128 tile; K scales share {0..7,16..23} or
{8..15,24..31} in each N32 tile. Each group spans all D128 channels. Scales
must match the stored codes; the tested preparation uses thread-MSE3 scales.
DELTA restores Q_block_mean @ (K_raw - K_mean).T before softmax, without
the attention scale (SM_SCALE is applied to both QK and DELTA here).

V is quantized after subtracting MU = mean_tokens(V_raw). EV is the mean
of raw-minus-reconstructed values, so it is added back in the epilogue:

    reconstructed = RN32(MU + RN32(VS * float32(V)))
    EV = mean_tokens(RN32(V_raw - reconstructed))

RN32 denotes a separate FP32 rounding step. Use the actual stored FP8 codes
and exclude padding from all means; EV is not reconstructed-minus-raw.

Padding must be initialized: zero Q/K/V payloads and finite positive QS/KS
through NP. Scales and V use unmasked loads; logical Q/K/DELTA/output remain
masked by N. Inputs must be finite, output storage must not alias inputs,
and the caller must validate shapes, dtypes, device, and int32 index bounds.

Launch on gfx1201 with grid (triton.cdiv(N, 128), B * H), num_warps=4,
num_stages=1, waves_per_eu=0, schedule_hint="none", and default FP fusion:

    _sage_attention_int4_fp8[(triton.cdiv(N, 128), B * H)](
        q4, k4, v8, qs, ks, vs, delta, q_original, k_mean, out, lse,
        error_mean, value_mean, N=N, NP=NP, H=H, QB=128,
        RETURN_LSE=lse is not None, BALANCED_MAX=False,
        num_warps=4, num_stages=1, waves_per_eu=0, schedule_hint="none",
    )

Pass None for Q_ORIG and LSE when RETURN_LSE=False. BALANCED_MAX selects
the row-maximum reduction: False is the measured 16K schedule, True the
balanced reduction measured at 114660 tokens. Both pack P early and use
the same quantization. INT4 accuracy is not equivalent to full precision
or INT8.
"""

import os

from triton._C.libtriton import ir
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.cdna3 import buffer_load
from triton.experimental.gluon.language.amd.rdna4 import wmma as _rdna4_wmma

_QK_LAYOUT = AMDWMMALayout(2, True, [[1, 0], [2, 0]], instr_shape=[16, 16, 32])
_PV_LAYOUT = AMDWMMALayout(2, True, [[1, 0], [2, 0]], instr_shape=[16, 16, 16])
_K_LOAD_LAYOUT = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
_V_LOAD_LAYOUT = gl.BlockedLayout([16, 1], [2, 16], [1, 4], [0, 1])

# Fixed vector-preserving LDS swizzles for N32, D128, four all-M waves.
# Both tiles are stored before K is consumed; V's fragment is loaded at PV.
_K_SHARED_LAYOUT = gl.SharedLinearLayout(
    [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 16],
        [0, 32],
        [1, 0],
        [2, 0],
        [4, 16],
        [8, 32],
        [16, 0],
        [0, 8],
    ],
    alignment=16,
)
_V_SHARED_LAYOUT = gl.SharedLinearLayout(
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


# Attention kernel. Hardware/compiler helpers follow below.
@gluon.jit
def _sage_attention_int4_fp8(
    Q,
    K,
    V,
    QS,
    KS,
    VS,
    DELTA,
    Q_ORIG,
    KM,
    O,
    LSE,
    EV,
    MU,
    N: gl.constexpr,
    NP: gl.constexpr,
    H: gl.constexpr,
    QB: gl.constexpr = 128,
    SM_SCALE: gl.constexpr = 128**-0.5,
    RETURN_LSE: gl.constexpr = False,
    P_SCALE: gl.constexpr = 448.0,
    BALANCED_MAX: gl.constexpr = False,
):
    qk_layout: gl.constexpr = _QK_LAYOUT
    pv_layout: gl.constexpr = _PV_LAYOUT
    BLOCK_M: gl.constexpr = 128
    BLOCK_N: gl.constexpr = 32
    HEAD_DIM: gl.constexpr = 128
    PACKED_DIM: gl.constexpr = 64
    LOG2_E: gl.constexpr = 1.4426950408889634
    LN_2: gl.constexpr = 0.6931471805599453
    FP8_MAX: gl.constexpr = 448.0
    gl.static_assert(
        NP >= N and NP % BLOCK_M == 0 and NP % BLOCK_N == 0,
        "prepared padding must cover all query/key tiles",
    )
    gl.static_assert(N > 0 and N <= 131072 and H > 0)
    gl.static_assert(QB == 128 or QB == 256)
    gl.static_assert(NP == ((N + QB - 1) // QB) * QB)
    gl.static_assert(SM_SCALE > 0 and P_SCALE >= 1 and P_SCALE <= FP8_MAX)
    gl.static_assert(
        O.dtype.element_ty == gl.bfloat16 or O.dtype.element_ty == gl.float16
    )
    query_tile = gl.program_id(0)
    batch_head = gl.program_id(1)

    # Query setup: Q stays in registers across every K/V iteration.
    # The INT4 helper preserves packed bytes through a compiler barrier.
    query_load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    query_rows_load = query_tile * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, query_load_layout)
    )
    packed_channels = gl.arange(
        0, PACKED_DIM, layout=gl.SliceLayout(0, query_load_layout)
    )
    packed_q = _load_tile(
        Q + batch_head * NP * PACKED_DIM,
        query_rows_load[:, None] * PACKED_DIM + packed_channels[None, :],
        query_rows_load[:, None] < N,
        0,
    )
    query_fragment = _prepare_i4_operand(packed_q, 0, qk_layout)
    query_rows = query_tile * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, qk_layout)
    )
    key_columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, qk_layout))
    # Rows r and r+64 share the scale stored at row r of each M128 tile.
    query_scale_rows = (query_rows // 128) * 128 + query_rows % 64
    # Preparation initializes the expanded scales through NP, including tails.
    query_scale = gl.load(QS + batch_head * NP + query_scale_rows)
    row_sum = gl.full((BLOCK_M,), 0.0, gl.float32, gl.SliceLayout(1, qk_layout))
    row_max = gl.full(
        (BLOCK_M,), float("-inf"), gl.float32, gl.SliceLayout(1, qk_layout)
    )
    output_acc = gl.full((BLOCK_M, HEAD_DIM), 0.0, gl.float32, pv_layout)

    # Streaming setup: prime K before the loop; V is loaded per iteration.
    key_load_layout: gl.constexpr = _K_LOAD_LAYOUT
    key_rows_load = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, key_load_layout))
    key_packed_channels = gl.arange(
        0, PACKED_DIM, layout=gl.SliceLayout(0, key_load_layout)
    )
    value_load_layout: gl.constexpr = _V_LOAD_LAYOUT
    value_rows = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, value_load_layout))
    value_channels = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, value_load_layout))
    key_limit = N
    packed_k = _load_tile(
        K + batch_head * NP * PACKED_DIM,
        key_rows_load[:, None] * PACKED_DIM + key_packed_channels[None, :],
        key_rows_load[:, None] < N,
        0,
    )
    for key_start in range(0, key_limit, BLOCK_N):
        # Stage K/V together, then compute INT4 QK.
        # Full vector V loads are valid because preparation zeros V through NP.
        loaded_values = _load_tile(
            V + batch_head * HEAD_DIM * NP,
            value_channels[None, :] * NP + key_start + value_rows[:, None],
            None,
            0.0,
        )
        qk_int = gl.full((BLOCK_M, BLOCK_N), 0, gl.int32, qk_layout)
        shared_k, shared_v = _stage_kv(packed_k, loaded_values)
        key_operand = shared_k.load(_packed_i4_layout(BLOCK_N, HEAD_DIM, 1, qk_layout))
        key_fragment = _prepare_i4_operand(key_operand, 1, qk_layout)
        qk_int = _wmma_i4_prepared(query_fragment, key_fragment, qk_int)
        # Prefetch packed K while the current tile's softmax and PV execute.
        next_packed_k = _load_tile(
            K + batch_head * NP * PACKED_DIM,
            (key_start + BLOCK_N + key_rows_load[:, None]) * PACKED_DIM
            + key_packed_channels[None, :],
            (key_start + BLOCK_N < key_limit)
            & (key_start + BLOCK_N + key_rows_load[:, None] < N),
            0,
        )

        # Dequantize QK and restore the Q-block-mean score correction.
        # Each N32 tile uses scale slot 0 for keys {0..7,16..23}, slot 8 otherwise.
        key_scale_rows = ((key_start + key_columns) // 32) * 32 + (
            ((key_start + key_columns) // 8) % 2
        ) * 8
        key_scale = gl.load(KS + batch_head * NP + key_scale_rows)
        scores = qk_int.to(gl.float32) * (query_scale[:, None] * key_scale[None, :])
        q_mean_correction = gl.load(
            DELTA
            + (batch_head * (NP // QB) + query_tile * BLOCK_M // QB) * NP
            + key_start
            + key_columns,
            key_start + key_columns < N,
            0,
        )
        scores = scores + q_mean_correction[None, :]
        scores = scores * (SM_SCALE * LOG2_E)
        valid_keys = key_start + key_columns[None, :] < N
        scores = gl.where(valid_keys, scores, float("-inf"))

        # Online softmax: scores/row_max are in base-2 units; P is unnormalized.
        if BALANCED_MAX:
            next_row_max = gl.maximum(row_max, _balanced_row_max(scores))
        else:
            next_row_max = gl.maximum(row_max, gl.max(scores, 1))
        unnormalized_p = gl.exp2(scores - next_row_max[:, None])
        rescale = gl.exp2(row_max - next_row_max)
        row_sum = row_sum * rescale + gl.sum(unnormalized_p, 1)

        # FP8 PV: sum unrounded P first, then pack before rescaling the accumulator.
        scaled_p = unnormalized_p * P_SCALE
        scaled_p = gl.minimum(scaled_p, FP8_MAX)
        packed_p = _to_fp8(scaled_p)
        rescale_pv = gl.convert_layout(rescale, gl.SliceLayout(1, pv_layout))
        output_acc = _gated_rescale(output_acc, rescale_pv)
        value_fragment = shared_v.load(gl.DotOperandLayout(1, pv_layout, 8))
        output_acc = _dot_fp8(packed_p, value_fragment, output_acc)
        row_max = next_row_max
        packed_k = next_packed_k

    # Output: four D32 register views limit the epilogue's live values.
    # Preserve the reciprocal and separate FP32 scale/mean/error steps before cast.
    denominator = gl.convert_layout(row_sum, gl.SliceLayout(1, pv_layout))
    output_rows = query_tile * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, pv_layout)
    )
    inv_denominator = 1.0 / (denominator * P_SCALE)
    output_quarters = _split_quarters(output_acc)
    for quarter in gl.static_range(0, 4):
        output_part = output_quarters[quarter]
        output_channels = quarter * 32 + gl.arange(
            0, 32, layout=gl.SliceLayout(0, pv_layout)
        )
        value_scale = gl.load(VS + batch_head * HEAD_DIM + output_channels)
        value_mean = gl.load(MU + batch_head * HEAD_DIM + output_channels)
        value_error = gl.load(EV + batch_head * HEAD_DIM + output_channels)
        normalized = output_part * inv_denominator[:, None]
        centered_output = _center_mul_f32(normalized, value_scale[None, :])
        with_mean = _center_add_f32(centered_output, value_mean[None, :])
        final_output = _center_add_f32(with_mean, value_error[None, :])
        output = final_output.to(O.dtype.element_ty)
        output_offsets = (
            batch_head // H * N * H + output_rows[:, None] * H + batch_head % H
        ) * HEAD_DIM + output_channels[None, :]
        gl.store(O + output_offsets, output, output_rows[:, None] < N)

    # Optional natural-log LSE restores the score offset removed by K centering.
    if RETURN_LSE:
        lse = (row_max + gl.log2(row_sum)) * LN_2
        original_query = gl.load(
            Q_ORIG
            + (batch_head // H * N * H + query_rows_load[:, None] * H + batch_head % H)
            * HEAD_DIM
            + gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, query_load_layout))[
                None, :
            ],
            query_rows_load[:, None] < N,
            0,
        ).to(gl.float32)
        mean_channels = gl.arange(
            0, HEAD_DIM, layout=gl.SliceLayout(0, query_load_layout)
        )
        key_mean = gl.load(KM + batch_head * HEAD_DIM + mean_channels)
        correction = gl.sum(original_query * key_mean[None, :], 1) * SM_SCALE
        lse = lse + gl.convert_layout(correction, gl.SliceLayout(1, qk_layout))
        gl.store(LSE + batch_head * N + query_rows, lse, query_rows < N)


# Vector loads and packed INT4 WMMA support.
@gluon.jit
def _load_tile(base, offsets, mask, other: gl.constexpr):
    # Keep this JIT boundary: inlining loads changes register scheduling.
    if mask is None:
        value = buffer_load(base, offsets)
    else:
        value = buffer_load(base, offsets, mask=mask, other=other)
    return value


def _get_int4_type(builder, context):
    # Types are owned by the compilation's MLIR context.  Cache on its builder,
    # never globally across compiler contexts.  parse_mlir_module accepts a
    # filename, so Linux memfd supplies the tiny type declaration in memory.
    if not hasattr(builder, "_rdna4_sage_i4_type"):
        fd = os.memfd_create("rdna4-sage-i4-type", flags=os.MFD_CLOEXEC)
        try:
            os.write(
                fd, b"module { tt.func private @sage_i4_type(%x: i4) { tt.return } }"
            )
            module = ir.parse_mlir_module(f"/proc/self/fd/{fd}", context)
            builder._rdna4_sage_i4_type = module.get_function(
                "sage_i4_type"
            ).type.param_types()[0]
        finally:
            os.close(fd)
    return builder._rdna4_sage_i4_type


@builtin
def _packed_i4_layout(rows, k, operand_index, acc_layout, _semantic=None):
    """Derive the byte layout by removing the native nibble register bit."""
    rows, k, operand_index, acc_layout = [
        x.value if isinstance(x, gl.constexpr) else x
        for x in (rows, k, operand_index, acc_layout)
    ]
    shape = [rows, k] if operand_index == 0 else [k, rows]
    layout = gl.DotOperandLayout(operand_index, acc_layout, 16)
    linear = _semantic.to_linear_layout(layout, shape).value
    k_dim = 1 - operand_index
    first = [0, 0]
    first[k_dim] = 1
    assert linear.reg_bases[0] == first

    def convert(bases):
        result = []
        for basis in bases:
            basis = list(basis)
            assert basis[k_dim] % 2 == 0
            basis[k_dim] //= 2
            if operand_index == 1:
                basis.reverse()
            result.append(basis)
        return result

    return gl.constexpr(
        gl.DistributedLinearLayout(
            convert(linear.reg_bases[1:]),
            convert(linear.lane_bases),
            convert(linear.warp_bases),
            convert(linear.block_bases),
            [rows, k // 2],
        )
    )


@builtin
def _wmma_i4_prepared(a, b, acc, _semantic=None):
    """Native signed INT4 dot on prepared nibble-valued INT8 operands.

    Use ``_prepare_i4_operand`` for A and B.  These Python-visible operands
    carry nibble values in i8; this builtin truncates to MLIR i4 before dot.
    The resulting machine instruction is v_wmma_i32_16x16x32_iu4.
    """
    layout = acc.type.layout
    if not isinstance(layout, AMDWMMALayout) or layout.version != 2:
        raise ValueError("INT4 WMMA requires an RDNA4 AMDWMMALayout")
    if layout.instr_shape != [16, 16, 32] or acc.dtype != gl.int32:
        raise ValueError(
            "INT4 WMMA requires K32 instruction shape and int32 accumulator"
        )
    for index, operand in enumerate((a, b)):
        if operand.dtype not in (gl.int8, gl.uint8):
            raise ValueError(
                "prepared nibble operands must have an 8-bit integer carrier dtype"
            )
        expected = gl.DotOperandLayout(index, layout, 16)
        if operand.type.layout != expected:
            raise ValueError(f"operand {index} must use {expected}")
    if a.type.shape[1] != b.type.shape[0] or a.type.shape[1] % 32:
        raise ValueError("INT4 WMMA K must agree and be a multiple of 32")
    if acc.type.shape != [a.type.shape[0], b.type.shape[1]]:
        raise ValueError("INT4 WMMA accumulator shape mismatch")
    builder = _semantic.builder
    i4 = _get_int4_type(builder, a.handle.get_context())
    native = []
    for operand in (a, b):
        native_type = builder.get_distributed_ty(
            i4, operand.type.shape, operand.type.layout._to_ir(builder)
        )
        native.append(builder.create_int_cast(operand.handle, native_type, False))
    result = builder.create_dot(
        native[0], native[1], acc.handle, ir.INPUT_PRECISION.IEEE, 0
    )
    return _semantic.tensor(result, acc.type)


@gluon.jit
def _prepare_i4_operand(packed, operand_index: gl.constexpr, acc_layout: gl.constexpr):
    """Prepare one K-packed uint8[M or N,K/2] matrix for native INT4 WMMA.

    Preparing Q outside the attention loop allows its converted fragment to
    be reused with every K tile.  No INT8 matrix instruction is involved.
    """
    gl.static_assert(packed.dtype == gl.uint8, "packed INT4 storage must be uint8")
    gl.static_assert(operand_index == 0 or operand_index == 1)
    M: gl.constexpr = packed.shape[0]
    K: gl.constexpr = packed.shape[1] * 2
    gl.static_assert(K % 32 == 0)
    # Exchange packed bytes, not expanded nibble carriers, between threads.
    packed = gl.convert_layout(
        packed, _packed_i4_layout(M, K, operand_index, acc_layout)
    )
    # Each pure empty asm is an identity on four bytes held in one VGPR.
    # It stops LLVM from folding the byte load into an illegal/shuffled i4
    # vector, which otherwise legalizes through private-memory spills.
    # A generic "=v" constraint rejects LLVM's <4 x i8> return type on this
    # toolchain; the explicit register accepts the same 32-bit carrier.
    # This can add register copies, but emits no nibble unpack instructions.
    packed = gl.inline_asm_elementwise(
        "",
        constraints="={v0},0",
        args=[packed],
        dtype=gl.uint8,
        is_pure=True,
        pack=4,
    )
    unpacked = gl.reshape(gl.join(packed & 15, packed >> 4), [M, K])
    if operand_index == 1:
        unpacked = gl.permute(unpacked, (1, 0))
    return gl.convert_layout(
        unpacked, gl.DotOperandLayout(operand_index, acc_layout, 16)
    )


# FP8 WMMA and native four-value conversion.
@gluon.jit
def _dot_fp8(a, b, acc):
    """Compute FP8 A[M,K] @ B[K,N] + FP32 acc with native RDNA4 WMMA."""
    gl.static_assert(acc.dtype == gl.float32)
    layout: gl.constexpr = acc.type.layout
    gl.static_assert(layout.version == 2)
    gl.static_assert(layout.instr_shape[0] == 16)
    gl.static_assert(layout.instr_shape[1] == 16)
    gl.static_assert(layout.instr_shape[2] == 16)
    a = gl.convert_layout(a, gl.DotOperandLayout(0, layout, 8))
    b = gl.convert_layout(b, gl.DotOperandLayout(1, layout, 8))
    return _rdna4_wmma(a, b, acc)


@gluon.jit
def _cvt4_scalar(x0, x1, x2, x3):
    # map_elementwise exposes scalar arguments, so this asm returns i32,
    # avoiding the <4 x i8> result type that rejects a generic VGPR.
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
def _to_fp8(x):
    """gfx1201 FP32 cast; finite [-448,448], four values per lane.

    Returns the same shape/layout with E4M3FN elements. The input layout's
    number of elements per thread must be divisible by four. RNE, signed
    zero, and E4M3FN subnormals are preserved; callers must clamp beforehand.
    """
    gl.static_assert(x.dtype == gl.float32, "native FP8 conversion requires FP32")
    # Keep the map result integer-typed: this compiler's map lowering tries
    # to pack scalar FP8 results before converting their LLVM carrier type.
    return gl.map_elementwise(_cvt4_scalar, x, pack=4)[0].to(
        gl.float8e4nv, bitcast=True
    )


# Shared-memory staging and gated accumulator rescaling.
@gluon.jit
def _stage_kv(k_packed, v_fp8):
    gl.static_assert(k_packed.dtype == gl.uint8)
    gl.static_assert(v_fp8.dtype == gl.float8e4nv)
    k_shared = gl.allocate_shared_memory(
        k_packed.dtype,
        k_packed.shape,
        _K_SHARED_LAYOUT,
        value=k_packed,
    )
    v_shared = gl.allocate_shared_memory(
        v_fp8.dtype,
        v_fp8.shape,
        _V_SHARED_LAYOUT,
        value=v_fp8,
    )
    return k_shared, v_shared


@builtin
def _row_group_size(acc, alpha, _semantic=None):
    """Prove the scalar rescale pack cannot cross query rows."""
    shape, layout = list(acc.type.shape), acc.type.layout
    if acc.dtype != gl.float32 or shape != [128, 128] or layout != _PV_LAYOUT:
        raise ValueError("gated rescale requires the fixed M128/D128 PV layout")
    if (
        alpha.dtype != gl.float32
        or list(alpha.type.shape) != [shape[0]]
        or alpha.type.layout != gl.SliceLayout(1, layout)
    ):
        raise ValueError(
            "gated rescale requires one FP32 alpha per row in SliceLayout(1, acc.layout)"
        )
    linear = _semantic.to_linear_layout(layout, shape).value
    group = 1
    for row_bit, _ in linear.reg_bases:
        if row_bit:
            break
        group *= 2
    if group < 8:
        raise ValueError(
            "gated rescale requires at least eight consecutive same-row registers"
        )
    return gl.constexpr(min(group, 64))


@gluon.jit
def _gated_rescale(acc, alpha):
    # The fixed PV layout gives each lane 64 consecutive values from one row.
    gl.static_assert(_row_group_size(acc, alpha) == 64)
    return gl.map_elementwise(_rescale64, acc, alpha[:, None], pack=64)[0]


# Output rounding and register-only channel views.
@gluon.jit
def _center_mul_f32(x, y):
    # Explicit FP32 boundaries prevent contraction with the two addbacks.
    return gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2;",
        constraints="=v,v,v",
        args=[x, y],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _center_add_f32(x, y):
    return gl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2;",
        constraints="=v,v,v",
        args=[x, y],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _split_channels(acc):
    # These layout-preserving register views do not exchange data between lanes.
    M: gl.constexpr = acc.type.shape[0]
    WIDTH: gl.constexpr = acc.type.shape[1]
    halves = gl.permute(gl.reshape(acc, (M, 2, WIDTH // 2)), (0, 2, 1))
    left, right = gl.split(halves)
    return (
        gl.convert_layout(left, acc.type.layout, assert_trivial=True),
        gl.convert_layout(right, acc.type.layout, assert_trivial=True),
    )


@gluon.jit
def _split_quarters(acc):
    left, right = _split_channels(acc)
    first, second = _split_channels(left)
    third, fourth = _split_channels(right)
    return first, second, third, fourth


# Optional balanced row-maximum reduction.
def _valid_max_layout(linear):
    return (
        linear.reg_bases == [[0, 1], [0, 2], [0, 4], [0, 16], [64, 0]]
        and linear.lane_bases == [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]]
        and linear.warp_bases == [[16, 0], [32, 0]]
        and linear.block_bases == []
    )


@builtin
def _assert_max_layout(layout, _semantic=None):
    layout = layout.value if isinstance(layout, gl.constexpr) else layout
    linear = _semantic.to_linear_layout(layout, [128, 32]).value
    if not _valid_max_layout(linear):
        raise ValueError(
            "balanced max requires exact long M128/N32 all-M register layout"
        )
    return gl.constexpr(True)


@gluon.jit
def _max3(a, b, c):
    return gl.maximum(gl.maximum(a, b), c)


@gluon.jit
def _max16_replicated(
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
):
    # Independent leaves expose a balanced local tree to instruction selection.
    t0 = _max3(x0, x1, x2)
    t1 = _max3(x3, x4, x5)
    t2 = _max3(x6, x7, x8)
    t3 = _max3(x9, x10, x11)
    t4 = _max3(x12, x13, x14)
    a = _max3(t0, t1, t2)
    b = _max3(t3, t4, x15)
    result = gl.maximum(a, b)
    return (
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
        result,
    )


@gluon.jit
def _balanced_row_max(x):
    gl.static_assert(x.dtype == gl.float32)
    gl.static_assert(x.shape == [128, 32])
    gl.static_assert(_assert_max_layout(x.type.layout))
    local = gl.map_elementwise(_max16_replicated, x, pack=16)[0]
    return gl.max(local, 1)


# Explicit scalar arguments are intentional: this compiler does not bind JIT
# varargs. Keeping each value branch-carried avoids eager multiplication when
# the row's rescale is exactly one. Do not split this pack into smaller groups.
# map_elementwise supplies 64 accumulator values and 64 broadcast scale values.
# _row_group_size proves that all c0..c63 belong to the same row and are equal,
# so only c0 is used; the other scale arguments are required by the map ABI.
@gluon.jit
def _rescale64(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    a16,
    a17,
    a18,
    a19,
    a20,
    a21,
    a22,
    a23,
    a24,
    a25,
    a26,
    a27,
    a28,
    a29,
    a30,
    a31,
    a32,
    a33,
    a34,
    a35,
    a36,
    a37,
    a38,
    a39,
    a40,
    a41,
    a42,
    a43,
    a44,
    a45,
    a46,
    a47,
    a48,
    a49,
    a50,
    a51,
    a52,
    a53,
    a54,
    a55,
    a56,
    a57,
    a58,
    a59,
    a60,
    a61,
    a62,
    a63,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    c16,
    c17,
    c18,
    c19,
    c20,
    c21,
    c22,
    c23,
    c24,
    c25,
    c26,
    c27,
    c28,
    c29,
    c30,
    c31,
    c32,
    c33,
    c34,
    c35,
    c36,
    c37,
    c38,
    c39,
    c40,
    c41,
    c42,
    c43,
    c44,
    c45,
    c46,
    c47,
    c48,
    c49,
    c50,
    c51,
    c52,
    c53,
    c54,
    c55,
    c56,
    c57,
    c58,
    c59,
    c60,
    c61,
    c62,
    c63,
):
    if c0 != 1.0:
        a0 = a0 * c0
        a1 = a1 * c0
        a2 = a2 * c0
        a3 = a3 * c0
        a4 = a4 * c0
        a5 = a5 * c0
        a6 = a6 * c0
        a7 = a7 * c0
        a8 = a8 * c0
        a9 = a9 * c0
        a10 = a10 * c0
        a11 = a11 * c0
        a12 = a12 * c0
        a13 = a13 * c0
        a14 = a14 * c0
        a15 = a15 * c0
        a16 = a16 * c0
        a17 = a17 * c0
        a18 = a18 * c0
        a19 = a19 * c0
        a20 = a20 * c0
        a21 = a21 * c0
        a22 = a22 * c0
        a23 = a23 * c0
        a24 = a24 * c0
        a25 = a25 * c0
        a26 = a26 * c0
        a27 = a27 * c0
        a28 = a28 * c0
        a29 = a29 * c0
        a30 = a30 * c0
        a31 = a31 * c0
        a32 = a32 * c0
        a33 = a33 * c0
        a34 = a34 * c0
        a35 = a35 * c0
        a36 = a36 * c0
        a37 = a37 * c0
        a38 = a38 * c0
        a39 = a39 * c0
        a40 = a40 * c0
        a41 = a41 * c0
        a42 = a42 * c0
        a43 = a43 * c0
        a44 = a44 * c0
        a45 = a45 * c0
        a46 = a46 * c0
        a47 = a47 * c0
        a48 = a48 * c0
        a49 = a49 * c0
        a50 = a50 * c0
        a51 = a51 * c0
        a52 = a52 * c0
        a53 = a53 * c0
        a54 = a54 * c0
        a55 = a55 * c0
        a56 = a56 * c0
        a57 = a57 * c0
        a58 = a58 * c0
        a59 = a59 * c0
        a60 = a60 * c0
        a61 = a61 * c0
        a62 = a62 * c0
        a63 = a63 * c0
    return (
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
        a8,
        a9,
        a10,
        a11,
        a12,
        a13,
        a14,
        a15,
        a16,
        a17,
        a18,
        a19,
        a20,
        a21,
        a22,
        a23,
        a24,
        a25,
        a26,
        a27,
        a28,
        a29,
        a30,
        a31,
        a32,
        a33,
        a34,
        a35,
        a36,
        a37,
        a38,
        a39,
        a40,
        a41,
        a42,
        a43,
        a44,
        a45,
        a46,
        a47,
        a48,
        a49,
        a50,
        a51,
        a52,
        a53,
        a54,
        a55,
        a56,
        a57,
        a58,
        a59,
        a60,
        a61,
        a62,
        a63,
    )
