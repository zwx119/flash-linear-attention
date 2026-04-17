# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang
"""
Test fused 🔥4+🔥5 pipeline v2 (Tensor Core, concurrent streams).
Compares against sequential execution for correctness.
"""
import pytest
import torch


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [128, 256])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("V", [64, 128])
def test_fused_h_o_pipeline_v2(B, T, H, K, V):
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    Hq = H  # no GQA for simplicity

    q = torch.randn(B, T, Hq, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hq, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)
    scale = K ** -0.5

    from fla.ops.delta_rule.fused_h_o_pipeline_v2 import (
        fused_chunk_h_o_pipeline,
        chunk_h_o_sequential,
    )

    # Reference: sequential
    o_ref, h_ref, vn_ref, _ = chunk_h_o_sequential(q, k, w, u, scale)

    # Test: pipeline
    o_test, h_test, vn_test, _ = fused_chunk_h_o_pipeline(q, k, w, u, scale)

    # Compare
    torch.testing.assert_close(h_test, h_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(vn_test, vn_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(o_test, o_ref, atol=1e-2, rtol=1e-2)
    print(f"✅ B={B}, T={T}, H={H}, K={K}, V={V} — pipeline matches sequential")


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("V", [128])
def test_pipeline_speedup(B, T, H, K, V):
    """Benchmark pipeline vs sequential to measure overlap benefit."""
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    Hq = H

    q = torch.randn(B, T, Hq, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hq, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)
    scale = K ** -0.5

    from fla.ops.delta_rule.fused_h_o_pipeline_v2 import (
        fused_chunk_h_o_pipeline,
        chunk_h_o_sequential,
    )

    # Warmup
    for _ in range(3):
        chunk_h_o_sequential(q, k, w, u, scale)
        fused_chunk_h_o_pipeline(q, k, w, u, scale)
    torch.cuda.synchronize()

    import time

    # Sequential
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        chunk_h_o_sequential(q, k, w, u, scale)
    torch.cuda.synchronize()
    t_seq = (time.perf_counter() - t0) / 20

    # Pipeline
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fused_chunk_h_o_pipeline(q, k, w, u, scale)
    torch.cuda.synchronize()
    t_pipe = (time.perf_counter() - t0) / 20

    speedup = t_seq / t_pipe
    print(f"Sequential: {t_seq*1000:.2f} ms, Pipeline: {t_pipe*1000:.2f} ms, Speedup: {speedup:.2f}x")
    print(f"  (B={B}, T={T}, H={H}, K={K}, V={V})")


if __name__ == "__main__":
    test_fused_h_o_pipeline_v2(B=1, T=128, H=4, K=128, V=128)
    test_pipeline_speedup(B=1, T=512, H=32, K=128, V=128)
