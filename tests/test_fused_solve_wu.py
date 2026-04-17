#!/usr/bin/env python3
"""
Test and benchmark script for the fused solve_tril + recompute_w_u kernel.

Compares the fused kernel (🔥2+🔥3 → 🔥2.3) against the original separate kernels
for both correctness and performance.

Usage:
    python test_fused_solve_wu.py              # correctness test only
    python test_fused_solve_wu.py --bench      # correctness + benchmark
    python test_fused_solve_wu.py --bench -n 100  # benchmark with 100 iterations
"""

import argparse
import time

import torch
import triton

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.delta_rule.fused_solve_wu import fused_solve_wu_fwd
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.utils.solve_tril import solve_tril


def generate_inputs(B=1, T=8192, H=32, K=128, V=128, dtype=torch.bfloat16, device='cuda'):
    """Generate random inputs matching the DeltaNet forward pass."""
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.1  # small positive beta
    return k, v, beta


def run_original(k, v, beta):
    """Run the original separate 🔥1 → 🔥2 → 🔥3 pipeline."""
    # 🔥1: chunk_scaled_dot_kkt
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
    # 🔥2: solve_tril
    Ai = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    # 🔥3: recompute_w_u
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
    return w, u, Ai, A


def run_fused(k, v, beta, A):
    """Run the fused 🔥2+🔥3 kernel (🔥1 output A is pre-computed)."""
    w, u, Ai = fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A)
    return w, u, Ai


def test_correctness(B=1, T=8192, H=32, K=128, V=128, dtype=torch.bfloat16):
    """Compare fused kernel output against original separate kernels."""
    print(f"=" * 70)
    print(f"Correctness Test: B={B}, T={T}, H={H}, K={K}, V={V}, dtype={dtype}")
    print(f"=" * 70)

    k, v, beta = generate_inputs(B, T, H, K, V, dtype)

    # Run original pipeline
    w_ref, u_ref, Ai_ref, A = run_original(k, v, beta)

    # Run fused kernel (reuse A from 🔥1)
    w_fused, u_fused, Ai_fused = run_fused(k, v, beta, A)

    # Compare Ai (L_inv)
    Ai_diff = (Ai_ref.float() - Ai_fused.float()).abs()
    Ai_max_err = Ai_diff.max().item()
    Ai_mean_err = Ai_diff.mean().item()
    print(f"  Ai (L_inv):  max_err={Ai_max_err:.6e}, mean_err={Ai_mean_err:.6e}", end="")
    print(f"  {'✅ PASS' if Ai_max_err < 1e-2 else '❌ FAIL'}")

    # Compare w
    w_diff = (w_ref.float() - w_fused.float()).abs()
    w_max_err = w_diff.max().item()
    w_mean_err = w_diff.mean().item()
    print(f"  W:           max_err={w_max_err:.6e}, mean_err={w_mean_err:.6e}", end="")
    print(f"  {'✅ PASS' if w_max_err < 1e-2 else '❌ FAIL'}")

    # Compare u
    u_diff = (u_ref.float() - u_fused.float()).abs()
    u_max_err = u_diff.max().item()
    u_mean_err = u_diff.mean().item()
    print(f"  U:           max_err={u_max_err:.6e}, mean_err={u_mean_err:.6e}", end="")
    print(f"  {'✅ PASS' if u_max_err < 1e-2 else '❌ FAIL'}")

    all_pass = Ai_max_err < 1e-2 and w_max_err < 1e-2 and u_max_err < 1e-2
    print(f"\n  Overall: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}\n")
    return all_pass


def benchmark(B=1, T=8192, H=32, K=128, V=128, dtype=torch.bfloat16, n_iter=50, warmup=10):
    """Benchmark fused vs. separate kernels."""
    print(f"=" * 70)
    print(f"Benchmark: B={B}, T={T}, H={H}, K={K}, V={V}, dtype={dtype}")
    print(f"  n_iter={n_iter}, warmup={warmup}")
    print(f"=" * 70)

    k, v, beta = generate_inputs(B, T, H, K, V, dtype)

    # Pre-compute A (🔥1) — shared by both paths
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)

    # --- Benchmark original: 🔥2 + 🔥3 (separate) ---
    for _ in range(warmup):
        Ai = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        Ai = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
    torch.cuda.synchronize()
    t_orig = (time.perf_counter() - t0) / n_iter * 1000  # ms

    # --- Benchmark fused: 🔥2.3 ---
    for _ in range(warmup):
        w, u, Ai = fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        w, u, Ai = fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / n_iter * 1000  # ms

    speedup = t_orig / t_fused if t_fused > 0 else float('inf')
    savings = t_orig - t_fused

    print(f"\n  Original (🔥2 + 🔥3):  {t_orig:.3f} ms")
    print(f"  Fused    (🔥2.3):      {t_fused:.3f} ms")
    print(f"  Savings:               {savings:.3f} ms ({speedup:.2f}×)")
    print()

    # --- Also benchmark with CUDA events for more accurate GPU timing ---
    print(f"  [CUDA Events timing]")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Original
    for _ in range(warmup):
        Ai = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(n_iter):
        Ai = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
    end_event.record()
    torch.cuda.synchronize()
    t_orig_cuda = start_event.elapsed_time(end_event) / n_iter

    # Fused
    for _ in range(warmup):
        w, u, Ai = fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(n_iter):
        w, u, Ai = fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A)
    end_event.record()
    torch.cuda.synchronize()
    t_fused_cuda = start_event.elapsed_time(end_event) / n_iter

    speedup_cuda = t_orig_cuda / t_fused_cuda if t_fused_cuda > 0 else float('inf')
    savings_cuda = t_orig_cuda - t_fused_cuda

    print(f"  Original (🔥2 + 🔥3):  {t_orig_cuda:.3f} ms")
    print(f"  Fused    (🔥2.3):      {t_fused_cuda:.3f} ms")
    print(f"  Savings:               {savings_cuda:.3f} ms ({speedup_cuda:.2f}×)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Test and benchmark fused solve_tril + recompute_w_u kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark after correctness test')
    parser.add_argument('-n', '--n-iter', type=int, default=50, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('-B', type=int, default=1, help='Batch size')
    parser.add_argument('-T', type=int, default=8192, help='Sequence length')
    parser.add_argument('-H', type=int, default=32, help='Number of heads')
    parser.add_argument('-K', type=int, default=128, help='Key dimension')
    parser.add_argument('-V', type=int, default=128, help='Value dimension')
    args = parser.parse_args()

    # Run correctness tests at multiple scales
    all_pass = True
    for T in [128, 512, 2048, args.T]:
        all_pass &= test_correctness(B=args.B, T=T, H=args.H, K=args.K, V=args.V)

    if not all_pass:
        print("❌ Correctness tests failed. Skipping benchmark.")
        return

    if args.bench:
        benchmark(
            B=args.B, T=args.T, H=args.H, K=args.K, V=args.V,
            n_iter=args.n_iter, warmup=args.warmup,
        )


if __name__ == '__main__':
    main()
