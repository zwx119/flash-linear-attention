"""
Benchmark: fused_solve_wu (🔥2+🔥3 fused) vs original separate kernels.

Usage:
    python benchmarks/bench_fused_solve_wu.py

Measures kernel-level latency for prepare_wy_repr_fwd with use_fused_kernel=True/False.
"""

import torch
import triton

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd, recompute_w_u_fwd
from fla.ops.delta_rule.fused_solve_wu import fused_solve_wu_fwd


def make_inputs(B, T, H, K, V, device='cuda', dtype=torch.bfloat16):
    """Create random inputs matching DeltaNet shapes."""
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).clamp(0.01, 0.99)
    return k, v, beta


def bench_original(k, v, beta, warmup=50, rep=200):
    """Benchmark original separate 🔥1 + 🔥2 + 🔥3."""
    # Warmup
    for _ in range(warmup):
        prepare_wy_repr_fwd(k, v, beta, cu_seqlens=None, use_fused_kernel=False)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        prepare_wy_repr_fwd(k, v, beta, cu_seqlens=None, use_fused_kernel=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def bench_fused(k, v, beta, warmup=50, rep=200):
    """Benchmark fused 🔥1 + 🔥2.3."""
    # Warmup
    for _ in range(warmup):
        prepare_wy_repr_fwd(k, v, beta, cu_seqlens=None, use_fused_kernel=True)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        prepare_wy_repr_fwd(k, v, beta, cu_seqlens=None, use_fused_kernel=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def bench_kernel_only(k, v, beta, warmup=50, rep=200):
    """Benchmark 🔥2+🔥3 only (excluding 🔥1), to isolate the fused kernel gain."""
    # Pre-compute A from 🔥1
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, cu_seqlens=None, chunk_size=64,
        output_dtype=torch.float32, chunk_indices=None,
    )

    # --- Original: 🔥2 (solve_tril) + 🔥3 (recompute_w_u) ---
    for _ in range(warmup):
        Ai = solve_tril(A=A, cu_seqlens=None, chunk_indices=None, output_dtype=k.dtype)
        recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        Ai = solve_tril(A=A, cu_seqlens=None, chunk_indices=None, output_dtype=k.dtype)
        recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
    end.record()
    torch.cuda.synchronize()
    t_orig = start.elapsed_time(end) / rep

    # --- Fused: 🔥2.3 (fused_solve_wu) ---
    for _ in range(warmup):
        fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=None)
    torch.cuda.synchronize()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(rep):
        fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=None)
    end2.record()
    torch.cuda.synchronize()
    t_fused = start2.elapsed_time(end2) / rep

    return t_orig, t_fused


def main():
    device = 'cuda'
    torch.manual_seed(42)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'='*80}")

    # --- Config space matching DeltaNet typical shapes ---
    configs = [
        # (B, T, H, K, V) — typical DeltaNet configs
        (1,  2048, 32, 128, 128),   # inference-like
        (2,  2048, 32, 128, 128),   # small batch
        (4,  2048, 32, 128, 128),   # medium batch
        (1,  4096, 32, 128, 128),   # longer seq
        (2,  4096, 32, 128, 128),
        (1,  8192, 32, 128, 128),   # long seq
        (2,  8192, 32, 128, 128),
        (1, 16384, 32, 128, 128),   # very long seq
        # 1.3B model (H=16, K=V=128)
        (2,  4096, 16, 128, 128),
        (4,  4096, 16, 128, 128),
        # 2.7B model (H=32, K=V=128)
        (2,  4096, 32, 128, 128),
        # 7B model (H=32, K=V=128)
        (1,  4096, 32, 128, 128),
    ]

    # De-dup
    seen = set()
    unique_configs = []
    for c in configs:
        if c not in seen:
            seen.add(c)
            unique_configs.append(c)

    print(f"\n{'B':>3} {'T':>6} {'H':>4} {'K':>4} {'V':>4} | "
          f"{'orig 🔥2+🔥3':>12} {'fused 🔥2.3':>12} {'speedup':>8} | "
          f"{'orig all':>10} {'fused all':>10} {'speedup':>8}")
    print(f"{'-'*3} {'-'*6} {'-'*4} {'-'*4} {'-'*4} | "
          f"{'-'*12} {'-'*12} {'-'*8} | "
          f"{'-'*10} {'-'*10} {'-'*8}")

    for B, T, H, K, V in unique_configs:
        try:
            k, v, beta = make_inputs(B, T, H, K, V, device=device)

            # Kernel-only benchmark (🔥2+🔥3 isolation)
            t_orig_kernel, t_fused_kernel = bench_kernel_only(k, v, beta)
            speedup_kernel = t_orig_kernel / t_fused_kernel

            # End-to-end benchmark (🔥1+🔥2+🔥3 combined)
            t_orig_all = bench_original(k, v, beta)
            t_fused_all = bench_fused(k, v, beta)
            speedup_all = t_orig_all / t_fused_all

            print(f"{B:>3} {T:>6} {H:>4} {K:>4} {V:>4} | "
                  f"{t_orig_kernel:>10.3f}ms {t_fused_kernel:>10.3f}ms {speedup_kernel:>7.2f}x | "
                  f"{t_orig_all:>8.3f}ms {t_fused_all:>8.3f}ms {speedup_all:>7.2f}x")

            del k, v, beta
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{B:>3} {T:>6} {H:>4} {K:>4} {V:>4} | OOM")
            torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("orig 🔥2+🔥3 = solve_tril + recompute_w_u (2 kernel launches)")
    print("fused 🔥2.3  = fused_solve_wu (1 kernel launch, no L_inv HBM round-trip)")
    print("orig all     = 🔥1 + 🔥2 + 🔥3 (3 kernel launches)")
    print("fused all    = 🔥1 + 🔥2.3 (2 kernel launches)")


if __name__ == '__main__':
    main()
