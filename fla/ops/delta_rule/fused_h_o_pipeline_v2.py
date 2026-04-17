# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang
# Fused 🔥4+🔥5 Pipeline — Tensor Core version
# Author: zwx119 (ByteDance)
#
# Strategy: minimal modification to original Triton kernels.
#   🔥4 (chunk_h):  + membar.gl + atomicAdd(sem, 1) after each chunk
#   🔥5 (chunk_o):  + spin-wait on sem before computation
#   Both kernels keep all tl.dot calls → Tensor Core matmul, identical to original.
#
# Launch: two concurrent CUDA streams.
#   🔥4 grid: (cdiv(V,BV), B*H) — few CTAs, sequential over NT chunks
#   🔥5 grid: (cdiv(V,BV), NT, B*H) — many CTAs, one per (v_slice, chunk, batch*head)
#   Spare SMs run 🔥5 CTAs while 🔥4 is still iterating → overlap.
#
# Memory ordering:
#   Producer: store h[t], store v_new[t] → membar.gl → atomicAdd(sem)
#   Consumer: spin on atomicAdd(sem,0) → membar.gl → load h[t], v_new[t]

import torch
import triton
import triton.language as tl

from fla.utils import check_shared_mem


# ═══════════════════════════════════════════════════════════════════════
#  Inline ASM helpers for cross-CTA synchronization
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _membar_gl():
    """__threadfence() equivalent — ensures prior stores visible to all SMs."""
    tl.inline_asm_elementwise(
        "membar.gl;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PRODUCER: 🔥4 chunk_h with semaphore signal
# ═══════════════════════════════════════════════════════════════════════
#
# Copied from fla/ops/common/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64
# with the following changes:
#   1. Removed: USE_G, USE_GK, IS_VARLEN, TRANSPOSE_STATE (delta_rule fwd doesn't use them)
#   2. Added: sem parameter + membar.gl + tl.atomic_add after each chunk
#   3. Everything else (tl.dot, pointer arithmetic, state management) is IDENTICAL.

@triton.jit(do_not_specialize=['T'])
def producer_kernel_h(
    k,
    v,        # this is u (the "new value" input)
    w,
    v_new,
    h,
    h0,
    ht,
    sem,      # [NT, B*H] int32 semaphore (NEW)
    T,
    H: tl.constexpr,
    Hq: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # State registers — identical to original
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # Pointer base offsets — identical to original
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # Load initial state — identical to original
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # Semaphore address for this (batch, head)
    # sem layout: [B*H, NT], so sem[i_bh, i_t] = sem + i_bh * NT + i_t
    i_bh = i_n * H + i_h

    # ── Main recurrence loop — identical to original, plus signal ──
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)

        # ①  Store h[t] — identical to original
        p_h1 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # ②  Compute v_new = u - W · state — identical to original (tl.dot = Tensor Core)
        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        # Store v_new — identical to original
        p_vn = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        # ③ === NEW: memory fence + signal semaphore ===
        #     This is the ONLY addition to the original kernel.
        #     Ensures h[t] and v_new[t] are visible to consumer CTAs on other SMs.
        _membar_gl()
        # sem layout: [B*H, NT], sem[i_bh, i_t] = sem + i_bh * NT + i_t
        tl.atomic_add(sem + i_bh * NT + i_t, 1)

        # ④  State update: h += k^T @ v_new — identical to original (tl.dot = Tensor Core)
        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    # Store final state — identical to original
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ═══════════════════════════════════════════════════════════════════════
#  CONSUMER: 🔥5 chunk_o with semaphore wait
# ═══════════════════════════════════════════════════════════════════════
#
# Copied from fla/ops/common/chunk_o.py::chunk_fwd_kernel_o
# with the following changes:
#   1. Removed: USE_G, USE_G_GAMMA, IS_VARLEN, TRANSPOSE_STATE
#   2. Added: sem + spin-wait + membar.gl before computation
#   3. Everything else (tl.dot, causal mask, output) is IDENTICAL.

@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def consumer_kernel_o(
    q,
    k,
    v,        # this is v_new
    h,
    o,
    sem,      # [B*H, NT] int32 semaphore (NEW)
    scale,
    T,
    H: tl.constexpr,
    Hq: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_PRODUCERS: tl.constexpr,  # number of producer CTAs per (batch, head)
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    NT = tl.cdiv(T, BT)
    bos, eos = i_b * T, i_b * T + T

    # === NEW: Spin-wait for all producers to finish chunk i_t ===
    i_bh_flat = i_b * H + i_h
    sem_addr = sem + i_bh_flat * NT + i_t
    # All threads execute this, but atomicAdd(ptr, 0) coalesces to one op per warp.
    while tl.atomic_add(sem_addr, 0) < NUM_PRODUCERS:
        pass
    _membar_gl()
    # === End of NEW section. Everything below is identical to original. ===

    # Offset calculation — identical to original
    q += (bos * Hq + i_h // (H // Hq)) * K
    k += (bos * Hq + i_h // (H // Hq)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_b * NT * H + i_t * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hq * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # Tensor Core matmul — identical to original
        b_o += tl.dot(b_q, b_h)    # [BT, BK] @ [BK, BV] → [BT, BV]
        b_A += tl.dot(b_q, b_k)    # [BT, BK] @ [BK, BT] → [BT, BT]

    # Causal mask — identical to original
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    # Final output — identical to original
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale   # Tensor Core!
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ═══════════════════════════════════════════════════════════════════════
#  PIPELINE LAUNCHER
# ═══════════════════════════════════════════════════════════════════════

def fused_chunk_h_o_pipeline(
    q: torch.Tensor,       # [B, T, Hq, K] bf16
    k: torch.Tensor,       # [B, T, Hq, K] bf16
    w: torch.Tensor,       # [B, T, H, K]  bf16
    u: torch.Tensor,       # [B, T, H, V]  bf16
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Fused 🔥4 + 🔥5 pipeline with Tensor Core and concurrent streams.

    Equivalent to:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, ...)
        o = chunk_fwd_o(q, k, v_new, h, scale=scale, ...)
    but overlaps 🔥5 with 🔥4 on spare SMs.
    """
    B, T, Hq, K = q.shape
    H = w.shape[2]
    V = u.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)

    # Allocate outputs — identical to original
    h = q.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)
    o = torch.empty_like(u)
    final_state = q.new_zeros(B, H, K, V, dtype=torch.float32) if output_final_state else None

    # Semaphore: [B*H, NT] int32, zero-initialized
    # sem[i_bh, i_t] counts how many producer V-slices have finished chunk i_t for this (batch, head)
    BH = B * H
    sem = torch.zeros(BH, NT, dtype=torch.int32, device=q.device)

    # Number of producer CTAs per (batch, head) = cdiv(V, BV_producer)
    # Fix BV for producer to ensure consumer knows the exact count for semaphore threshold.
    if check_shared_mem('ada'):
        BV_PROD = 64
    else:
        BV_PROD = 32

    NUM_PRODUCERS = triton.cdiv(V, BV_PROD)

    # Producer grid: (cdiv(V, BV), B*H)
    grid_prod = (NUM_PRODUCERS, B * H)

    # Consumer grid: (cdiv(V, BV_cons), NT, B*H)
    # BV_cons is autotuned; grid uses meta['BV']
    def grid_cons(meta):
        return (triton.cdiv(V, meta['BV']), NT, B * H)

    # Launch on concurrent streams
    stream_prod = torch.cuda.Stream()
    stream_cons = torch.cuda.Stream()

    with torch.cuda.stream(stream_prod):
        producer_kernel_h[grid_prod](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            h=h,
            h0=initial_state,
            ht=final_state,
            sem=sem,
            T=T,
            H=H,
            Hq=Hq,
            K=K,
            V=V,
            BT=BT,
            BV=BV_PROD,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
        )

    with torch.cuda.stream(stream_cons):
        consumer_kernel_o[grid_cons](
            q=q,
            k=k,
            v=v_new,
            h=h,
            o=o,
            sem=sem,
            scale=scale,
            T=T,
            H=H,
            Hq=Hq,
            K=K,
            V=V,
            BT=BT,
            NUM_PRODUCERS=NUM_PRODUCERS,
        )

    # Sync both streams back to the current stream
    torch.cuda.current_stream().wait_stream(stream_prod)
    torch.cuda.current_stream().wait_stream(stream_cons)

    return o, h, v_new, final_state


# ═══════════════════════════════════════════════════════════════════════
#  FALLBACK: sequential execution (identical to original)
# ═══════════════════════════════════════════════════════════════════════

def chunk_h_o_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """Fallback: run original 🔥4 then 🔥5 sequentially."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=None, scale=scale,
    )
    return o, h, v_new, final_state
