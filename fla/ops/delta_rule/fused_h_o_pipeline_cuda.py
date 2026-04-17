# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Fused 🔥4+🔥5 Pipeline — CUDA version Python wrapper
# Author: zwx119 (ByteDance)
#
# JIT-compiles the CUDA persistent kernel and provides a drop-in replacement
# for the sequential chunk_gated_delta_rule_fwd_h + chunk_fwd_o call.

import os
import torch
import triton

from fla.ops.delta_rule.sm_occupancy import analyze_pipeline_feasibility
from fla.utils import check_shared_mem

# ─── JIT compilation ───
_cuda_module = None


def _get_cuda_module():
    """Lazy JIT-compile the CUDA kernel on first use."""
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc')
    _cuda_module = load(
        name='fused_h_o_pipeline_cuda',
        sources=[os.path.join(csrc_dir, 'fused_h_o_pipeline.cu')],
        extra_include_paths=[csrc_dir],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '-gencode=arch=compute_80,code=sm_80',   # A100
            '-gencode=arch=compute_89,code=sm_89',   # Ada
            '-gencode=arch=compute_90,code=sm_90',   # H100
        ],
        verbose=False,
    )
    return _cuda_module


def fused_h_o_pipeline_cuda(
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
    Fused 🔥4+🔥5 pipeline using CUDA persistent kernel.

    Advantages over Triton version:
      - Guaranteed __threadfence() semantics
      - Proper volatile loads for spin-wait (no atomic_add hack)
      - __launch_bounds__ for register pressure control
      - Separate __noinline__ __device__ functions for producer/consumer

    Falls back to sequential Triton kernels if:
      - Not enough spare SMs for pipeline
      - CUDA compilation fails
      - Input dtype is not bf16

    Args:
        q: [B, T, H, K] bf16
        k: [B, T, H, K] bf16
        w: [B, T, H, K] bf16 (from 🔥2+🔥3)
        u: [B, T, H, V] bf16 (from 🔥2+🔥3)
        scale: attention scale (default: K^{-0.5})
        initial_state: not yet supported in pipeline mode
        output_final_state: not yet supported in pipeline mode

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

    # ─── Feasibility check ───
    analysis = analyze_pipeline_feasibility(B, T, H, K, V, BT)

    # Conditions for CUDA pipeline
    can_pipeline = (
        analysis['pipeline_feasible']
        and q.dtype == torch.bfloat16
        and initial_state is None
        and not output_final_state
    )

    if not can_pipeline:
        return _fallback_sequential(q, k, w, u, scale, initial_state,
                                     output_final_state, BT)

    # ─── Try CUDA pipeline ───
    try:
        cuda_mod = _get_cuda_module()
    except Exception as e:
        import warnings
        warnings.warn(f"CUDA JIT compilation failed, falling back to sequential: {e}")
        return _fallback_sequential(q, k, w, u, scale, initial_state,
                                     output_final_state, BT)

    # ─── Allocate outputs ───
    h = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=q.device)
    v_new = torch.empty_like(u)
    o = torch.empty_like(u)
    sem = torch.zeros(NT * B * H, dtype=torch.int32, device=q.device)

    # ─── Choose tile sizes ───
    if check_shared_mem('ada'):
        BV_PROD = 64
    else:
        BV_PROD = 32

    if check_shared_mem('hopper'):
        BV_CONS = 128
    elif check_shared_mem('ada'):
        BV_CONS = 64
    else:
        BV_CONS = 32
    # Clamp BV_CONS to V
    BV_CONS = min(BV_CONS, V)

    # ─── Launch ───
    cuda_mod.fused_h_o_pipeline_launch(
        q.contiguous(), k.contiguous(), w.contiguous(), u.contiguous(),
        h, v_new, o, sem,
        scale, BV_PROD, BV_CONS,
    )

    final_state = None
    return o, h, v_new, final_state


def _fallback_sequential(q, k, w, u, scale, initial_state, output_final_state, BT):
    """Fall back to original sequential 🔥4 → 🔥5."""
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
