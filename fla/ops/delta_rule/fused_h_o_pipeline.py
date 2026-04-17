# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Fused 🔥4+🔥5 Pipeline: chunk_h (producer) || chunk_o (consumer)
# Author: zwx119 (ByteDance)
#
# When 🔥4 uses fewer CTAs than available SMs, the spare SMs sit idle.
# This kernel fuses 🔥4 and 🔥5 into a single persistent launch:
#
#   Producer CTAs (first fire4_ctas): run 🔥4's sequential recurrence.
#     After computing h[t] and v_new[t], signal consumer via atomic.
#
#   Consumer CTAs (remaining): run 🔥5's per-chunk output computation.
#     Spin-wait on semaphore, then compute o[t] = q·h[t] + tril(q·k^T)·v_new[t].
#
# Cross-CTA synchronization:
#   - Producer: store h[t], v_new[t] → threadfence → atomic_add(sem[t], 1)
#   - Consumer: spin on atomic_load(sem[t]) until all producer slices ready
#   - threadfence via tl.inline_asm_elementwise("membar.gl;", ...)
#
# Applicable when: spare_sms >= fire5_ctas_per_chunk (see sm_occupancy.py)

import torch
import triton
import triton.language as tl

from fla.ops.delta_rule.sm_occupancy import analyze_pipeline_feasibility
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import check_shared_mem


def _threadfence():
    """Emit a GPU-wide memory fence (membar.gl) via inline PTX."""
    # Returns a dummy value; the side effect is the fence instruction.
    return tl.inline_asm_elementwise(
        asm="membar.gl; mov.u32 $0, 0;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=['T'])
def _producer_body(
    # 🔥4 logic: sequential recurrence h[t] → v_new[t] → update state
    k, w, u, h, v_new, semaphore,
    bos, T, i_v, i_h, i_nh,
    H: tl.constexpr,
    Hq: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    """Producer CTA: runs 🔥4 recurrence, signals consumer after each chunk."""
    NT = tl.cdiv(T, BT)

    # offset pointers
    h_base = h + (bos // T * NT * H + i_h).to(tl.int64) * K * V
    u_base = u + (bos * H + i_h).to(tl.int64) * V
    k_base = k + (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
    w_base = w + (bos * H + i_h).to(tl.int64) * K
    v_new_base = v_new + (bos * H + i_h).to(tl.int64) * V

    # Initialize state: b_h [K_block, BV] or [BV, K_block]
    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    for i_t in range(NT):
        i_t_i64 = i_t.to(tl.int64)

        # ── Store h[t] ──
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h_base + i_t_i64 * H * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h1 = tl.make_block_ptr(h_base + i_t_i64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(h_base + i_t_i64 * H * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h2 = tl.make_block_ptr(h_base + i_t_i64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        # ── Compute v_new[t] = u[t] - w[t] · h[t] ──
        p_w = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w2 = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))
            else:
                b_v += tl.dot(b_w2, b_h2.to(b_w2.dtype))

        p_u = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v

        # Store v_new[t]
        p_vn = tl.make_block_ptr(v_new_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        # ── threadfence + signal consumer ──
        _threadfence()
        # semaphore layout: [NT, N*H], producer CTA (i_v, i_nh) adds 1
        # consumer waits until sem[t, i_nh] == num_producer_per_head
        tl.atomic_add(semaphore + i_t * (tl.cdiv(T, BT).to(tl.int64) * 0 + 1) + i_nh, 1,
                       sem='relaxed', scope='gpu')
        # Simpler: flatten semaphore as [NT * N*H]
        # Actually let's use a simpler layout
        # We'll signal per (t, nh): sem[t * NH + nh]

        # ── Update state: h += k^T · v_new ──
        b_v_cast = b_v.to(k.dtype.element_ty)
        p_k1 = tl.make_block_ptr(k_base, (K, T), (1, Hq * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k1, b_v_cast))
        else:
            b_h1 += tl.dot(b_k1, b_v_cast)
        if K > 64:
            p_k2 = tl.make_block_ptr(k_base, (K, T), (1, Hq * K), (64, i_t * BT), (64, BT), (0, 1))
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k2, b_v_cast))
            else:
                b_h2 += tl.dot(b_k2, b_v_cast)


@triton.jit(do_not_specialize=['T'])
def _consumer_body(
    # 🔥5 logic: compute o[t] = q·h[t]*scale + tril(q·k^T)*v_new[t]*scale
    q, k, v_new, h, o, semaphore,
    bos, T, i_v, i_bh, i_b, i_h,
    num_producer_per_head,
    scale,
    H: tl.constexpr,
    Hq: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    """Consumer CTA: waits for producer, then computes one chunk of output."""
    NT = tl.cdiv(T, BT)

    # Each consumer CTA processes chunks in a strided pattern
    # Consumer CTAs are indexed by (i_v, i_t, i_bh) but we flatten them
    # For simplicity, each consumer processes chunks round-robin
    # Total consumer work items: cdiv(V, BV) * NT * B*H
    # But we have fewer consumer CTAs, so each does multiple items

    # offset pointers
    q_base = q + (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
    k_base = k + (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
    v_new_base = v_new + (bos * H + i_h).to(tl.int64) * V
    o_base = o + (bos * H + i_h).to(tl.int64) * V
    h_base = h + (i_b * NT * H + i_h).to(tl.int64) * K * V
    NH = tl.cdiv(T, 1)  # placeholder - we pass NH explicitly

    for i_t in range(NT):
        # ── Spin-wait for producer to finish chunk t ──
        # sem[t * (B*H) + i_bh] counts how many producer V-slices are done
        sem_addr = semaphore + i_t.to(tl.int64) * 1 + 0  # simplified
        # We need all V-slice producers for this head to have written h[t] and v_new[t]
        while tl.atomic_add(sem_addr, 0, sem='relaxed', scope='gpu') < num_producer_per_head:
            pass  # spin

        _threadfence()

        # ── Load h[t] and compute o[t] ──
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q_base, (T, K), (Hq * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_base, (K, T), (1, Hq * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            if TRANSPOSE_STATE:
                p_h = tl.make_block_ptr(h_base + i_t.to(tl.int64) * H * K * V, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_base + i_t.to(tl.int64) * H * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h_tile = tl.load(p_h, boundary_check=(0, 1))

            if TRANSPOSE_STATE:
                b_o += tl.dot(b_q, tl.trans(b_h_tile))
            else:
                b_o += tl.dot(b_q, b_h_tile)
            b_A += tl.dot(b_q, b_k)

        # Apply causal mask
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        # Load v_new[t] and compute final output
        p_v = tl.make_block_ptr(v_new_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ─── Main fused kernel: dispatches producer vs consumer by program_id ───

@triton.jit(do_not_specialize=['T', 'num_producer_ctas'])
def fused_h_o_pipeline_kernel(
    # 🔥4 inputs
    k, w, u,
    # 🔥5 inputs
    q,
    # outputs
    h, v_new, o,
    # sync
    semaphore,
    # dims
    T, B: tl.constexpr, H: tl.constexpr, Hq: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr,
    # producer config
    num_producer_ctas,
    num_producer_per_head,  # cdiv(V, BV_producer)
    BV_PROD: tl.constexpr,
    # consumer config
    BK_CONS: tl.constexpr,
    BV_CONS: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    scale,
):
    pid = tl.program_id(0)

    if pid < num_producer_ctas:
        # ── Producer CTA: 🔥4 logic ──
        # Grid mapping: pid → (i_v, i_nh) where i_v = pid // (B*H), i_nh = pid % (B*H)
        NH = B * H
        i_v = pid // NH
        i_nh = pid % NH
        i_n = i_nh // H
        i_h = i_nh % H
        bos = i_n * T

        NT = tl.cdiv(T, BT)

        # Initialize state
        if TRANSPOSE_STATE:
            b_h1 = tl.zeros([BV_PROD, 64], dtype=tl.float32)
        else:
            b_h1 = tl.zeros([64, BV_PROD], dtype=tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                b_h2 = tl.zeros([BV_PROD, 64], dtype=tl.float32)
            else:
                b_h2 = tl.zeros([64, BV_PROD], dtype=tl.float32)

        # Offset pointers
        h_off = (i_n * NT * H + i_h).to(tl.int64) * K * V
        u_off = (bos * H + i_h).to(tl.int64) * V
        k_off = (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
        w_off = (bos * H + i_h).to(tl.int64) * K
        vn_off = (bos * H + i_h).to(tl.int64) * V

        for i_t in range(NT):
            i_t_i64 = i_t.to(tl.int64)

            # Store h[t]
            if TRANSPOSE_STATE:
                p_h1 = tl.make_block_ptr(h + h_off + i_t_i64 * H * K * V, (V, K), (K, 1), (i_v * BV_PROD, 0), (BV_PROD, 64), (1, 0))
            else:
                p_h1 = tl.make_block_ptr(h + h_off + i_t_i64 * H * K * V, (K, V), (V, 1), (0, i_v * BV_PROD), (64, BV_PROD), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                if TRANSPOSE_STATE:
                    p_h2 = tl.make_block_ptr(h + h_off + i_t_i64 * H * K * V, (V, K), (K, 1), (i_v * BV_PROD, 64), (BV_PROD, 64), (1, 0))
                else:
                    p_h2 = tl.make_block_ptr(h + h_off + i_t_i64 * H * K * V, (K, V), (V, 1), (64, i_v * BV_PROD), (64, BV_PROD), (1, 0))
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

            # v_new[t] = u[t] - w[t] · h_state
            p_w = tl.make_block_ptr(w + w_off, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
            else:
                b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w2 = tl.make_block_ptr(w + w_off, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w2 = tl.load(p_w2, boundary_check=(0, 1))
                if TRANSPOSE_STATE:
                    b_v += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))
                else:
                    b_v += tl.dot(b_w2, b_h2.to(b_w2.dtype))

            p_u = tl.make_block_ptr(u + u_off, (T, V), (H * V, 1), (i_t * BT, i_v * BV_PROD), (BT, BV_PROD), (1, 0))
            b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v

            p_vn = tl.make_block_ptr(v_new + vn_off, (T, V), (H * V, 1), (i_t * BT, i_v * BV_PROD), (BT, BV_PROD), (1, 0))
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

            # Fence + signal
            _threadfence()
            # sem[i_t, i_nh]: each producer V-slice increments by 1
            tl.atomic_add(semaphore + i_t * (B * H) + i_nh, 1, sem='relaxed', scope='gpu')

            # Update state: h += k^T · v_new
            b_v_cast = b_v.to(k.dtype.element_ty)
            p_k1 = tl.make_block_ptr(k + k_off, (K, T), (1, Hq * K), (0, i_t * BT), (64, BT), (0, 1))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h1 += tl.trans(tl.dot(b_k1, b_v_cast))
            else:
                b_h1 += tl.dot(b_k1, b_v_cast)
            if K > 64:
                p_k2 = tl.make_block_ptr(k + k_off, (K, T), (1, Hq * K), (64, i_t * BT), (64, BT), (0, 1))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                if TRANSPOSE_STATE:
                    b_h2 += tl.trans(tl.dot(b_k2, b_v_cast))
                else:
                    b_h2 += tl.dot(b_k2, b_v_cast)

    else:
        # ── Consumer CTA: 🔥5 logic ──
        # Consumer pid mapping: (pid - num_producer_ctas) → (i_v, i_t, i_bh)
        consumer_id = pid - num_producer_ctas
        NT = tl.cdiv(T, BT)
        NV_cons = tl.cdiv(V, BV_CONS)
        total_items = NV_cons * NT * B * H

        # Each consumer CTA processes items round-robin
        # Total consumer CTAs = grid_size - num_producer_ctas
        # We don't know grid_size in kernel, so compute from available info
        # For now, each consumer handles items strided by num_consumer_ctas
        # But we don't have num_consumer_ctas as constexpr...
        # Instead, unroll: consumer_id maps to one (i_v, i_t, i_bh)
        # If num_consumer >= total_items, 1:1 mapping. Otherwise round-robin.

        # 1:1 mapping (simplest; launch enough consumer CTAs)
        if consumer_id < total_items:
            i_v = consumer_id // (NT * B * H)
            remainder = consumer_id % (NT * B * H)
            i_t = remainder // (B * H)
            i_bh = remainder % (B * H)
            i_b = i_bh // H
            i_h = i_bh % H

            bos = i_b * T

            # Spin-wait for producer
            sem_idx = i_t * (B * H) + i_bh
            while tl.atomic_add(semaphore + sem_idx, 0, sem='relaxed', scope='gpu') < num_producer_per_head:
                pass
            _threadfence()

            # Compute o[t]
            q_off = (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
            k_off = (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
            vn_off = (bos * H + i_h).to(tl.int64) * V
            o_off = (bos * H + i_h).to(tl.int64) * V
            h_off = (i_b * NT * H + i_h).to(tl.int64) * K * V

            b_o = tl.zeros([BT, BV_CONS], dtype=tl.float32)
            b_A = tl.zeros([BT, BT], dtype=tl.float32)

            for i_k in range(tl.cdiv(K, BK_CONS)):
                p_q = tl.make_block_ptr(q + q_off, (T, K), (Hq * K, 1), (i_t * BT, i_k * BK_CONS), (BT, BK_CONS), (1, 0))
                p_k = tl.make_block_ptr(k + k_off, (K, T), (1, Hq * K), (i_k * BK_CONS, i_t * BT), (BK_CONS, BT), (0, 1))
                if TRANSPOSE_STATE:
                    p_h = tl.make_block_ptr(h + h_off + i_t.to(tl.int64) * H * K * V, (V, K), (K, 1), (i_v * BV_CONS, i_k * BK_CONS), (BV_CONS, BK_CONS), (1, 0))
                else:
                    p_h = tl.make_block_ptr(h + h_off + i_t.to(tl.int64) * H * K * V, (K, V), (V, 1), (i_k * BK_CONS, i_v * BV_CONS), (BK_CONS, BV_CONS), (1, 0))

                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h_tile = tl.load(p_h, boundary_check=(0, 1))

                if TRANSPOSE_STATE:
                    b_o += tl.dot(b_q, tl.trans(b_h_tile))
                else:
                    b_o += tl.dot(b_q, b_h_tile)
                b_A += tl.dot(b_q, b_k)

            # Causal mask
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
            b_A = tl.where(m_A, b_A, 0)

            p_v = tl.make_block_ptr(v_new + vn_off, (T, V), (H * V, 1), (i_t * BT, i_v * BV_CONS), (BT, BV_CONS), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale

            p_o = tl.make_block_ptr(o + o_off, (T, V), (H * V, 1), (i_t * BT, i_v * BV_CONS), (BT, BV_CONS), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fused_h_o_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Fused 🔥4+🔥5 pipeline: overlaps chunk_h (producer) with chunk_o (consumer).

    Uses a persistent kernel with cross-CTA atomic signaling.
    Falls back to sequential execution if SM count is insufficient.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K] (Hq may differ from H for GQA)
        w: [B, T, H, K] from 🔥2+🔥3
        u: [B, T, H, V] from 🔥2+🔥3
        scale: attention scale (default: K^{-0.5})
        initial_state: [B, H, K, V] optional initial hidden state
        output_final_state: whether to return final state

    Returns:
        (o, h, v_new, final_state)
    """
    B, T, Hq, K = q.shape
    H = w.shape[2]
    V = u.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    if scale is None:
        scale = K ** -0.5

    # ─── Check feasibility ───
    analysis = analyze_pipeline_feasibility(B, T, H, K, V, BT)
    if not analysis['pipeline_feasible']:
        # Fall back to sequential execution
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla.ops.common.chunk_o import chunk_fwd_o

        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=BT,
        )
        o = chunk_fwd_o(
            q=q, k=k, v=v_new, h=h, g=None,
            scale=scale, chunk_size=BT,
        )
        return o, h, v_new, final_state

    # ─── Pipeline execution ───
    TRANSPOSE_STATE = False  # standard layout

    # Producer config (🔥4)
    if check_shared_mem('ada'):
        BV_PROD = 64
    else:
        BV_PROD = 32
    num_producer_per_head = triton.cdiv(V, BV_PROD)  # V-slices per (n, h)
    num_producer_ctas = num_producer_per_head * B * H

    # Consumer config (🔥5)
    if check_shared_mem('hopper'):
        BK_CONS, BV_CONS = 128, 128
    elif check_shared_mem('ada'):
        BK_CONS, BV_CONS = 64, 64
    else:
        BK_CONS, BV_CONS = 32, 32
    NV_cons = triton.cdiv(V, BV_CONS)
    num_consumer_ctas = NV_cons * NT * B * H

    # Allocate outputs
    h = q.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)
    o = torch.empty_like(u)

    # Semaphore: [NT, B*H], zero-initialized
    semaphore = torch.zeros(NT * B * H, dtype=torch.int32, device=q.device)

    total_ctas = num_producer_ctas + num_consumer_ctas

    fused_h_o_pipeline_kernel[(total_ctas,)](
        k=k, w=w, u=u, q=q,
        h=h, v_new=v_new, o=o,
        semaphore=semaphore,
        T=T, B=B, H=H, Hq=Hq, K=K, V=V, BT=BT,
        num_producer_ctas=num_producer_ctas,
        num_producer_per_head=num_producer_per_head,
        BV_PROD=BV_PROD,
        BK_CONS=BK_CONS,
        BV_CONS=BV_CONS,
        TRANSPOSE_STATE=TRANSPOSE_STATE,
        scale=scale,
        num_warps=4,
        num_stages=2,
    )

    final_state = None  # TODO: extract from last h state if needed

    return o, h, v_new, final_state
