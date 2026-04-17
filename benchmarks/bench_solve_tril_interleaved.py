"""
Benchmark: solve_tril CC-TC interleaved vs original.

Tests ONLY 🔥2 (solve_tril) — the CC-TC interleaving optimization.
W/U (🔥3) is untouched and uses the same original big-dot kernel.

Usage: python benchmarks/bench_solve_tril_interleaved.py
"""

import torch
import triton

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.index import prepare_chunk_indices


def make_inputs(B, T, H, K, device='cuda', dtype=torch.bfloat16):
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).clamp(0.01, 0.99)
    # Compute A from 🔥1
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, cu_seqlens=None, chunk_size=64,
        output_dtype=torch.float32, chunk_indices=None,
    )
    return A


def bench_solve_tril(A, kernel_fn, warmup=100, rep=500):
    BT = A.shape[-1]
    B, T, H = A.shape[:3]
    NT = triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=torch.bfloat16)

    for _ in range(warmup):
        kernel_fn[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=None, chunk_indices=None,
            T=T, H=H, BT=BT, USE_TMA=False,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel_fn[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=None, chunk_indices=None,
            T=T, H=H, BT=BT, USE_TMA=False,
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def main():
    from fla.ops.utils.solve_tril import merge_16x16_to_64x64_inverse_kernel
    from fla.ops.utils.solve_tril_interleaved import merge_16x16_to_64x64_interleaved_kernel

    device = 'cuda'
    torch.manual_seed(42)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Benchmark: solve_tril (🔥2) only — CC-TC interleaved vs original")
    print(f"{'='*80}")

    configs = [
        (1,  2048, 32, 128),
        (2,  2048, 32, 128),
        (4,  2048, 32, 128),
        (1,  4096, 32, 128),
        (2,  4096, 32, 128),
        (1,  8192, 32, 128),
        (2,  8192, 32, 128),
        (1, 16384, 32, 128),
        (2,  4096, 16, 128),
        (4,  4096, 16, 128),
    ]

    seen = set()
    unique = [c for c in configs if c not in seen and not seen.add(c)]

    print(f"\n{'B':>3} {'T':>6} {'H':>4} {'K':>4} | "
          f"{'original':>10} {'interleaved':>12} {'speedup':>8}")
    print("-" * 60)

    for B, T, H, K in unique:
        try:
            A = make_inputs(B, T, H, K, device=device)
            t_orig = bench_solve_tril(A, merge_16x16_to_64x64_inverse_kernel)
            t_inter = bench_solve_tril(A, merge_16x16_to_64x64_interleaved_kernel)
            print(f"{B:>3} {T:>6} {H:>4} {K:>4} | "
                  f"{t_orig:>8.4f}ms {t_inter:>10.4f}ms {t_orig/t_inter:>7.3f}x")
            del A
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{B:>3} {T:>6} {H:>4} {K:>4} | ERROR: {e}")
            torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("original    = CC1→CC2→CC3→CC4→TC(all 16 dots)")
    print("interleaved = CC1→CC2→TC(2)→CC3→TC(5)→CC4→TC(9)")
    print("On Hopper (wgmma async): 7 TC dots in steps 2+3 overlap with CC3+CC4")


if __name__ == '__main__':
    main()
