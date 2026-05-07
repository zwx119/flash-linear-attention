#!/usr/bin/env python3
"""Full DeltaNet chunk forward benchmark for H100 kernel profiling.

This runs `fla.ops.delta_rule.chunk_delta_rule`, so the profile includes the
full forward operator chain:

  1. chunk_scaled_dot_kkt_fwd_kernel
  2. solve_tril / merge_16x16_to_64x64_inverse_kernel
  3. recompute_w_u_fwd_kernel
  4. chunk_gated_delta_rule_fwd_kernel_h_blockdim64
  5. chunk_fwd_kernel_o

Use `--solve-wu-impl original|fused|hopper|overlap` to compare the WY section.
The default is `original`, matching the fastest measured H100 path.
"""

from __future__ import annotations

import argparse
import os
import statistics
from collections.abc import Callable

import torch


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
    normalize_qk: bool,
    beta_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    q = torch.randn(batch, seq, heads, key_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq, heads, key_dim, dtype=dtype, device=device)
    if normalize_qk:
        q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(dtype)
        k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(dtype)
    v = torch.randn(batch, seq, heads, value_dim, dtype=dtype, device=device)
    beta = torch.rand(batch, seq, heads, dtype=torch.float32, device=device) * beta_scale
    return q, k, v, beta


def time_cuda_events(
    label: str,
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor | None]],
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
            out, final_state = fn()
            # Keep outputs live through the timing region.
            if final_state is not None:
                out = out + final_state.flatten()[0].to(out.dtype)
            if profile:
                torch.cuda.nvtx.range_pop()
        end.record()
        if profile:
            torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_iter
        times.append(ms)
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {label:<12} repeat={repeat:<2} time_ms={ms:.4f} peak_gb={peak_gb:.3f}")
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
    parser.add_argument("--solve-wu-impl", choices=["original", "fused", "hopper", "overlap"], default="original")
    parser.add_argument("--batch", "-B", type=int, default=1)
    parser.add_argument("--seq", "-T", type=int, default=32768)
    parser.add_argument("--heads", "-H", type=int, default=32)
    parser.add_argument("--key-dim", "-K", type=int, default=80)
    parser.add_argument("--value-dim", "-V", type=int, default=80)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--beta-scale", type=float, default=0.1)
    parser.add_argument("--normalize-qk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true", help="Emit NVTX ranges around measured loops")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    # wy_fast reads this at import time, so set it before importing chunk_delta_rule.
    os.environ["FLA_SOLVE_WU_IMPL"] = args.solve_wu_impl
    from fla.ops.delta_rule import chunk_delta_rule

    print_device()
    print("Shape:")
    print(
        f"  B={args.batch} T={args.seq} H={args.heads} "
        f"K={args.key_dim} V={args.value_dim} dtype={args.dtype}"
    )
    print(f"  warmup={args.warmup} n_iter={args.n_iter} repeats={args.repeats}")
    print(f"  normalize_qk={args.normalize_qk} beta_scale={args.beta_scale}")
    print(f"  solve_wu_impl={args.solve_wu_impl}")

    q, k, v, beta = generate_inputs(
        batch=args.batch,
        seq=args.seq,
        heads=args.heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        dtype=_dtype(args.dtype),
        normalize_qk=args.normalize_qk,
        beta_scale=args.beta_scale,
    )

    scale = args.key_dim**-0.5

    def run_once() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )

    print("Benchmark:")
    times = time_cuda_events(
        f"deltanet_{args.solve_wu_impl}",
        run_once,
        warmup=args.warmup,
        n_iter=args.n_iter,
        repeats=args.repeats,
        profile=args.profile,
    )

    print("Summary:")
    summarize(args.solve_wu_impl, times)


if __name__ == "__main__":
    main()
