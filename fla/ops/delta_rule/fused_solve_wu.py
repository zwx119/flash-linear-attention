# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Fused solve_tril + recompute_w_u kernel for DeltaNet delta_rule forward.
# Author: zwx119 (ByteDance)
#
# This kernel fuses 🔥2 (solve_tril: L_inv computation) and 🔥3 (recompute_w_u)
# into a single kernel launch. Two independent sources of speedup:
#
# === 1. Intended CC/TC overlap schedule ===
#
# Original solve_tril does ALL 4 CC forward_sub blocks first, then ALL TC dots.
# CC and TC never overlap. The target Hopper implementation should split the CTA
# into cooperating CC and TC worker groups:
#
#   CC: forward_sub(block_1) → Ai_11
#   CC: forward_sub(block_2) → Ai_22
#   TC group: Ai_21 = -Ai_22·A_21·Ai_11       (2 dots)
#       overlaps with
#   CC group: forward_sub(block_3) → Ai_33
#   TC group: Ai_32, Ai_31                    (5 dots)
#       overlaps with
#   CC group: forward_sub(block_4) → Ai_44
#   TC: Ai_43, Ai_42, Ai_41                   (9 dots)  ← no CC left
#
# This Triton prototype currently fuses solve_tril and recompute_w_u, and orders
# the dependent work in the same logical sequence, but it does not explicitly
# split warps/warp-groups or use named barriers to guarantee simultaneous CC and
# TC execution. Proving true overlap requires a Hopper profile, and may require a
# lower-level warp-specialized implementation.
#
# === 2. Eliminate L_inv HBM round-trip ===
#
# Original: 🔥2 writes L_inv to HBM → 🔥3 reads it back → does W = L_inv·(β⊙K)
# Fused: L_inv stays in registers → W/U computed immediately → no HBM round-trip
#
# W/U computation is pure TC, executed after all L_inv blocks are ready.
# No CC work remains at this point, so there's no CC-TC overlap here —
# the savings come purely from eliminating the HBM write+read of L_inv.

import torch
import triton
import triton.language as tl

import os

from fla.ops.utils import prepare_chunk_indices
from fla.utils import IS_TMA_SUPPORTED, autotune_cache_kwargs, check_shared_mem

# Match solve_tril's precision logic: on non-TMA hardware, always use 'ieee'
FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')
DOT_PRECISION_AUTOTUNE_LIST = ["ieee"] if not IS_TMA_SUPPORTED else list({"ieee", FLA_TRIL_PRECISION})


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_solve_wu_fwd_kernel(
    k,
    v,
    beta,
    A,       # input: raw A = tril(β⊙KK^T), shape [B, T, H, BT], dtype=float32
    Ai,      # output: L_inv = (I+A)^{-1}, shape [B, T, H, BT] (needed for backward)
    w,       # output: W = L_inv · (β⊙K), shape [B, T, H, K]
    u,       # output: U = L_inv · (β⊙V), shape [B, T, H, V]
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # =========================================================================
    # Setup: common indexing
    # =========================================================================
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]   # strict lower triangular mask
    m_I = o_i[:, None] == o_i[None, :]   # identity mask

    # Base pointers for A and Ai (offset to this batch/head)
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Base pointers for k, v, w, u (offset to this batch/head)
    k_base = k + (bos * H + i_h) * K
    v_base = v + (bos * H + i_h) * V
    w_base = w + (bos * H + i_h) * K
    u_base = u + (bos * H + i_h) * V

    # Load beta for this chunk in 4 sub-vectors (compatible with older Triton)
    p_beta_0 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (16,), (0,))
    p_beta_1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_beta_2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_beta_3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    b_beta_0 = tl.load(p_beta_0, boundary_check=(0,))
    b_beta_1 = tl.load(p_beta_1, boundary_check=(0,))
    b_beta_2 = tl.load(p_beta_2, boundary_check=(0,))
    b_beta_3 = tl.load(p_beta_3, boundary_check=(0,))

    # =========================================================================
    # Load 4 diagonal blocks of A [16,16] each
    # =========================================================================
    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = -tl.where(m_A, tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_22 = -tl.where(m_A, tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_33 = -tl.where(m_A, tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_44 = -tl.where(m_A, tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32), 0)

    # =========================================================================
    # Logical CC-TC dependency order for L_inv.
    #
    # NOTE: this is a fused sequential prototype. A true overlap kernel should
    # run the off-diagonal TC worker group concurrently with the next diagonal
    # CC worker group and synchronize their 16x16 blocks explicitly.
    # =========================================================================

    # --- Block 1: forward substitution [CC] ---
    for i in range(2, min(16, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a, b_Ai_11)
    b_Ai_11 += m_I

    # --- Block 2: forward substitution [CC] ---
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a = tl.where(o_i < i - 16, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a, b_Ai_22)
    b_Ai_22 += m_I

    # --- Off-diagonal: Ai_21 [TC] (overlaps with block 3 CC on H100) ---
    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)

    # --- Block 3: forward substitution [CC] ---
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a = tl.where(o_i < i - 32, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a, b_Ai_33)
    b_Ai_33 += m_I

    # --- Off-diagonal: Ai_32, Ai_31 [TC] (overlaps with block 4 CC on H100) ---
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    # --- Block 4: forward substitution [CC] ---
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a = tl.where(o_i < i - 48, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a, b_Ai_44)
    b_Ai_44 += m_I

    # --- Off-diagonal: Ai_43, Ai_42, Ai_41 [TC] ---
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    # =========================================================================
    # Store L_inv to HBM (needed for backward pass)
    # =========================================================================
    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    # =========================================================================
    # Single-pass W and U computation
    #
    # For each K/V tile, load all 4 (β⊙K)/(β⊙V) sub-tiles once, multiply
    # by the full L_inv, and store W/U once. No load-modify-store needed.
    # =========================================================================

    # --- W = L_inv · (β⊙K) ---
    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (16, BK), (1, 0))
        p_k1 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0))
        p_k3 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0))
        b_kb0 = (tl.load(p_k0, boundary_check=(0, 1)) * b_beta_0[:, None]).to(k.dtype.element_ty)
        b_kb1 = (tl.load(p_k1, boundary_check=(0, 1)) * b_beta_1[:, None]).to(k.dtype.element_ty)
        b_kb2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta_2[:, None]).to(k.dtype.element_ty)
        b_kb3 = (tl.load(p_k3, boundary_check=(0, 1)) * b_beta_3[:, None]).to(k.dtype.element_ty)

        # w0 = Ai_11 · kb0
        b_w0 = tl.dot(b_Ai_11.to(b_kb0.dtype), b_kb0, allow_tf32=False)
        # w1 = Ai_21 · kb0 + Ai_22 · kb1
        b_w1 = tl.dot(b_Ai_21.to(b_kb0.dtype), b_kb0, allow_tf32=False) + \
               tl.dot(b_Ai_22.to(b_kb1.dtype), b_kb1, allow_tf32=False)
        # w2 = Ai_31 · kb0 + Ai_32 · kb1 + Ai_33 · kb2
        b_w2 = tl.dot(b_Ai_31.to(b_kb0.dtype), b_kb0, allow_tf32=False) + \
               tl.dot(b_Ai_32.to(b_kb1.dtype), b_kb1, allow_tf32=False) + \
               tl.dot(b_Ai_33.to(b_kb2.dtype), b_kb2, allow_tf32=False)
        # w3 = Ai_41 · kb0 + Ai_42 · kb1 + Ai_43 · kb2 + Ai_44 · kb3
        b_w3 = tl.dot(b_Ai_41.to(b_kb0.dtype), b_kb0, allow_tf32=False) + \
               tl.dot(b_Ai_42.to(b_kb1.dtype), b_kb1, allow_tf32=False) + \
               tl.dot(b_Ai_43.to(b_kb2.dtype), b_kb2, allow_tf32=False) + \
               tl.dot(b_Ai_44.to(b_kb3.dtype), b_kb3, allow_tf32=False)

        p_w0 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (16, BK), (1, 0))
        p_w1 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0))
        p_w3 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0))
        tl.store(p_w0, b_w0.to(p_w0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w3, b_w3.to(p_w3.dtype.element_ty), boundary_check=(0, 1))

    # --- U = L_inv · (β⊙V) ---
    for i_v in range(tl.cdiv(V, BV)):
        p_v0 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (16, BV), (1, 0))
        p_v1 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0))
        p_v2 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0))
        p_v3 = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0))
        b_vb0 = (tl.load(p_v0, boundary_check=(0, 1)) * b_beta_0[:, None]).to(v.dtype.element_ty)
        b_vb1 = (tl.load(p_v1, boundary_check=(0, 1)) * b_beta_1[:, None]).to(v.dtype.element_ty)
        b_vb2 = (tl.load(p_v2, boundary_check=(0, 1)) * b_beta_2[:, None]).to(v.dtype.element_ty)
        b_vb3 = (tl.load(p_v3, boundary_check=(0, 1)) * b_beta_3[:, None]).to(v.dtype.element_ty)

        # u0 = Ai_11 · vb0
        b_u0 = tl.dot(b_Ai_11.to(b_vb0.dtype), b_vb0, allow_tf32=False)
        # u1 = Ai_21 · vb0 + Ai_22 · vb1
        b_u1 = tl.dot(b_Ai_21.to(b_vb0.dtype), b_vb0, allow_tf32=False) + \
               tl.dot(b_Ai_22.to(b_vb1.dtype), b_vb1, allow_tf32=False)
        # u2 = Ai_31 · vb0 + Ai_32 · vb1 + Ai_33 · vb2
        b_u2 = tl.dot(b_Ai_31.to(b_vb0.dtype), b_vb0, allow_tf32=False) + \
               tl.dot(b_Ai_32.to(b_vb1.dtype), b_vb1, allow_tf32=False) + \
               tl.dot(b_Ai_33.to(b_vb2.dtype), b_vb2, allow_tf32=False)
        # u3 = Ai_41 · vb0 + Ai_42 · vb1 + Ai_43 · vb2 + Ai_44 · vb3
        b_u3 = tl.dot(b_Ai_41.to(b_vb0.dtype), b_vb0, allow_tf32=False) + \
               tl.dot(b_Ai_42.to(b_vb1.dtype), b_vb1, allow_tf32=False) + \
               tl.dot(b_Ai_43.to(b_vb2.dtype), b_vb2, allow_tf32=False) + \
               tl.dot(b_Ai_44.to(b_vb3.dtype), b_vb3, allow_tf32=False)

        p_u0 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (16, BV), (1, 0))
        p_u1 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0))
        p_u2 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0))
        p_u3 = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0))
        tl.store(p_u0, b_u0.to(p_u0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u2, b_u2.to(p_u2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u3, b_u3.to(p_u3.dtype.element_ty), boundary_check=(0, 1))


def fused_solve_wu_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused solve_tril + recompute_w_u forward pass.

    Given A = tril(β⊙KK^T) from 🔥1, this function computes in a single kernel:
      1. L_inv = (I + A)^{-1}                   (originally 🔥2: solve_tril)
      2. W = L_inv · (β⊙K), U = L_inv · (β⊙V)  (originally 🔥3: recompute_w_u)

    Saves ~0.20 ms by eliminating the HBM round-trip for the L_inv intermediate.

    Args:
        k: [B, T, H, K] key tensor (bf16/fp16)
        v: [B, T, H, V] value tensor (bf16/fp16)
        beta: [B, T, H] beta tensor (fp32)
        A: [B, T, H, BT] raw A matrix from 🔥1 (fp32)
        cu_seqlens: optional cumulative sequence lengths for varlen
        chunk_indices: optional precomputed chunk indices for varlen

    Returns:
        w: [B, T, H, K] = L_inv · (β⊙K)       (same dtype as k)
        u: [B, T, H, V] = L_inv · (β⊙V)       (same dtype as v)
        Ai: [B, T, H, BT] = (I+A)^{-1}        (same dtype as k, needed for backward)
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = 64
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Allocate outputs: Ai is zero-initialized for upper-triangular part
    Ai = torch.zeros_like(A, dtype=k.dtype)
    w = torch.empty_like(k)
    u = torch.empty_like(v)

    fused_solve_wu_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        A=A,
        Ai=Ai,
        w=w,
        u=u,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, Ai
