# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# H100 warp-specialized triangular solve prototype.

from __future__ import annotations

import os
import warnings

import torch

from fla.ops.utils.solve_tril import solve_tril
from fla.utils import IS_NVIDIA_HOPPER

_cuda_module = None


def _get_cuda_module():
    """Lazy JIT-compile the CUDA extension on first use."""
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
    _cuda_module = load(
        name="overlap_solve_tril_cuda",
        sources=[os.path.join(csrc_dir, "overlap_solve_tril.cu")],
        extra_include_paths=[csrc_dir],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-gencode=arch=compute_90,code=sm_90",
        ],
        verbose=False,
    )
    return _cuda_module


def overlap_solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute (I + A)^-1 with an H100 warp-specialized solve prototype.

    The CUDA prototype intentionally handles only the common DeltaNet benchmark
    path: dense non-varlen chunks, BT=64, fp32 A, bf16 output, and sequence
    length divisible by 64. Everything else falls back to the original FLA
    solve_tril so callers can opt in safely.
    """
    output_dtype = A.dtype if output_dtype is None else output_dtype
    can_use_cuda = (
        IS_NVIDIA_HOPPER
        and A.is_cuda
        and A.dtype == torch.float32
        and output_dtype == torch.bfloat16
        and A.ndim == 4
        and A.shape[-1] == 64
        and A.shape[1] % 64 == 0
        and cu_seqlens is None
        and chunk_indices is None
    )
    if not can_use_cuda:
        return solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=output_dtype,
        )

    try:
        cuda_mod = _get_cuda_module()
    except Exception as exc:
        warnings.warn(f"overlap_solve_tril CUDA JIT failed; falling back to solve_tril: {exc}", stacklevel=2)
        return solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=output_dtype,
        )

    return cuda_mod.overlap_solve_tril_fwd(A.contiguous())
