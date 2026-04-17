#!/usr/bin/env python3
"""Debug script to locate the exact source of error in fused_solve_wu."""

import torch
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.delta_rule.fused_solve_wu import fused_solve_wu_fwd
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.utils.solve_tril import solve_tril


def debug_test(B=1, T=64, H=1, K=128, V=128):
    """Minimal test: single chunk, single head."""
    print(f"\n{'='*60}")
    print(f"Debug: B={B}, T={T}, H={H}, K={K}, V={V}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
    beta = torch.rand(B, T, H, dtype=torch.float32, device='cuda') * 0.1

    # 🔥1
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)

    # Original: 🔥2 + 🔥3
    Ai_ref = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w_ref, u_ref = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai_ref, cu_seqlens=None)

    # Fused: 🔥2.3
    w_fused, u_fused, Ai_fused = fused_solve_wu_fwd(k=k, v=v, beta=beta, A=A)

    # --- Debug Ai ---
    diff = (Ai_ref.float() - Ai_fused.float()).abs()
    max_err = diff.max().item()
    idx = (diff == diff.max()).nonzero(as_tuple=False)[0]
    b, t, h, bt = idx.tolist()
    print(f"\nAi max_err={max_err:.6e} at [{b},{t},{h},{bt}]")
    print(f"  ref={Ai_ref[b,t,h,bt].item():.6f}, fused={Ai_fused[b,t,h,bt].item():.6f}")

    # Show which 16x16 block this falls in
    chunk_idx = t // 64
    local_row = t % 64
    local_col = bt
    block_row = local_row // 16
    block_col = local_col // 16
    print(f"  chunk={chunk_idx}, local=({local_row},{local_col}), block=({block_row},{block_col})")
    if block_row < block_col:
        print(f"  ** This is UPPER TRIANGULAR — should be 0!")
    elif block_row == block_col:
        print(f"  ** This is a DIAGONAL block")
    else:
        print(f"  ** This is a LOWER off-diagonal block")

    # Show top-10 errors
    flat_diff = diff.flatten()
    topk = torch.topk(flat_diff, min(10, flat_diff.numel()))
    print(f"\nTop-10 Ai errors:")
    for val, flat_idx in zip(topk.values, topk.indices):
        idx4 = []
        remaining = flat_idx.item()
        for s in [T * H * 64, H * 64, 64]:
            idx4.append(remaining // s)
            remaining = remaining % s
        idx4.append(remaining)
        b, t, h, bt = idx4
        chunk_idx = t // 64
        lr, lc = t % 64, bt
        br, bc = lr // 16, lc // 16
        ref_val = Ai_ref[b, t, h, bt].item()
        fused_val = Ai_fused[b, t, h, bt].item()
        pos = "UPPER" if br < bc else ("DIAG" if br == bc else "LOWER")
        print(f"  err={val.item():.4f} [{b},{t},{h},{bt}] chunk={chunk_idx} "
              f"block=({br},{bc}) {pos} ref={ref_val:.4f} fused={fused_val:.4f}")

    # --- Debug W ---
    w_diff = (w_ref.float() - w_fused.float()).abs()
    w_max_err = w_diff.max().item()
    print(f"\nW max_err={w_max_err:.6e}")

    # --- Debug U ---
    u_diff = (u_ref.float() - u_fused.float()).abs()
    u_max_err = u_diff.max().item()
    print(f"U max_err={u_max_err:.6e}")


if __name__ == '__main__':
    # Clear triton cache first
    import shutil, os
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.exists(cache_dir):
        print(f"Clearing Triton cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    # Single chunk, single head (minimal)
    debug_test(B=1, T=64, H=1, K=128, V=128)
    # Single chunk, multi head
    debug_test(B=1, T=64, H=32, K=128, V=128)
    # Multi chunk
    debug_test(B=1, T=128, H=32, K=128, V=128)
