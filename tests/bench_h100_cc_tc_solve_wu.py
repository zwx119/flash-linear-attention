#!/usr/bin/env python3
"""H100 benchmark/profiling entry for DeltaNet solve_wu CC/TC experiments.

This isolates the forward WY preparation hot section:

  original: solve_tril(A) + recompute_w_u(k, v, beta, Ai)
  fused:    fused_solve_wu(k, v, beta, A)

The input A = tril(beta * K K^T) is precomputed once so both paths measure only
the solve+WU region. Use Nsight Compute/System around this script to check
whether the fused Triton kernel improves wall time and whether H100 reports
concurrent tensor-pipe and non-tensor-pipe activity.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.delta_rule.fused_solve_wu import fused_solve_wu_fwd
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.utils.solve_tril import solve_tril


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def generate_inputs(
    batch: int,
    seq: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    k = torch.randn(batch, seq, heads, key_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq, heads, value_dim, dtype=dtype, device=device)
    beta = torch.rand(batch, seq, heads, dtype=torch.float32, device=device) * 0.1
    return k, v, beta


def make_a(k: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        cu_seqlens=None,
        chunk_size=64,
        output_dtype=torch.float32,
    )


def run_original(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ai = solve_tril(A=a_raw, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=ai, cu_seqlens=None)
    return w, u, ai


def run_fused(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fused_solve_wu_fwd(k=k, v=v, beta=beta, A=a_raw)


def max_mean_err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


def check_correctness(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a_raw: torch.Tensor,
    atol: float,
) -> None:
    print("Correctness:")
    w_ref, u_ref, ai_ref = run_original(k, v, beta, a_raw)
    w_fused, u_fused, ai_fused = run_fused(k, v, beta, a_raw)

    passed = True
    for name, ref, got in (
        ("Ai", ai_ref, ai_fused),
        ("W", w_ref, w_fused),
        ("U", u_ref, u_fused),
    ):
        max_err, mean_err = max_mean_err(ref, got)
        ok = max_err <= atol
        passed = passed and ok
        status = "OK" if ok else "FAIL"
        print(f"  {name:<2} max_err={max_err:.6e} mean_err={mean_err:.6e} [{status}]")

    if not passed:
        raise SystemExit(f"correctness failed with atol={atol}")


def time_cuda_events(
    label: str,
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    warmup: int,
    n_iter: int,
    repeats: int,
    profile: bool,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for repeat in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if profile:
            torch.cuda.nvtx.range_push(f"{label}_repeat_{repeat}")
        start.record()
        for i in range(n_iter):
            if profile:
                torch.cuda.nvtx.range_push(f"{label}_iter_{i}")
            fn()
            if profile:
                torch.cuda.nvtx.range_pop()
        end.record()
        if profile:
            torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_iter
        times.append(ms)
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {label:<8} repeat={repeat:<2} time_ms={ms:.4f} peak_gb={peak_gb:.3f}")
    return times


def summarize(label: str, times: list[float]) -> None:
    mean = statistics.mean(times)
    stdev = statistics.pstdev(times) if len(times) > 1 else 0.0
    median = statistics.median(times)
    print(f"RESULT mode={label} mean_ms={mean:.4f} median_ms={median:.4f} std_ms={stdev:.4f}")


def print_device() -> None:
    props = torch.cuda.get_device_properties(0)
    print("Device:")
    print(f"  name={props.name}")
    print(f"  capability=sm_{props.major}{props.minor}")
    print(f"  total_memory_gb={props.total_memory / 1024**3:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["original", "fused", "both"], default="both")
    parser.add_argument("--batch", "-B", type=int, default=1)
    parser.add_argument("--seq", "-T", type=int, default=8192)
    parser.add_argument("--heads", "-H", type=int, default=32)
    parser.add_argument("--key-dim", "-K", type=int, default=128)
    parser.add_argument("--value-dim", "-V", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Emit NVTX ranges around measured loops")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print_device()
    print("Shape:")
    print(
        f"  B={args.batch} T={args.seq} H={args.heads} "
        f"K={args.key_dim} V={args.value_dim} dtype={args.dtype}"
    )
    print(f"  warmup={args.warmup} n_iter={args.n_iter} repeats={args.repeats}")

    k, v, beta = generate_inputs(
        batch=args.batch,
        seq=args.seq,
        heads=args.heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        dtype=_dtype(args.dtype),
    )
    a_raw = make_a(k, beta)
    torch.cuda.synchronize()

    if not args.skip_correctness:
        check_correctness(k, v, beta, a_raw, args.atol)

    print("Benchmark:")
    results: dict[str, list[float]] = {}
    if args.mode in ("original", "both"):
        results["original"] = time_cuda_events(
            "original",
            lambda: run_original(k, v, beta, a_raw),
            warmup=args.warmup,
            n_iter=args.n_iter,
            repeats=args.repeats,
            profile=args.profile,
        )
    if args.mode in ("fused", "both"):
        results["fused"] = time_cuda_events(
            "fused",
            lambda: run_fused(k, v, beta, a_raw),
            warmup=args.warmup,
            n_iter=args.n_iter,
            repeats=args.repeats,
            profile=args.profile,
        )

    print("Summary:")
    for label, times in results.items():
        summarize(label, times)
    if "original" in results and "fused" in results:
        orig = statistics.mean(results["original"])
        fused = statistics.mean(results["fused"])
        print(f"RESULT speedup_fused_vs_original={orig / fused:.4f} saved_ms={orig - fused:.4f}")


if __name__ == "__main__":
    main()
