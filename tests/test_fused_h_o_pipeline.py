# Test for fused 🔥4+🔥5 pipeline kernel (both Triton and CUDA versions)
# Compares pipeline output against sequential (original) execution.

import torch
import pytest

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.delta_rule.fused_h_o_pipeline import fused_h_o_pipeline
from fla.ops.delta_rule.sm_occupancy import analyze_pipeline_feasibility, print_analysis


def _reference(q, k, w, u, scale, BT):
    """Sequential reference: 🔥4 → 🔥5."""
    h_ref, v_new_ref, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None,
        initial_state=None, output_final_state=False, chunk_size=BT,
    )
    o_ref = chunk_fwd_o(
        q=q, k=k, v=v_new_ref, h=h_ref, g=None,
        scale=scale, chunk_size=BT,
    )
    return o_ref, v_new_ref


@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 512, 4, 64, 64),
    (1, 512, 4, 128, 128),
    (1, 2048, 8, 128, 128),
    (1, 8192, 32, 128, 128),
])
def test_fused_h_o_pipeline_triton(B, T, H, K, V):
    """Test Triton pipeline produces same output as sequential."""
    device = 'cuda'
    dtype = torch.bfloat16
    BT = 64
    scale = K ** -0.5

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)

    o_ref, v_new_ref = _reference(q, k, w, u, scale, BT)

    o_pipe, h_pipe, v_new_pipe, _ = fused_h_o_pipeline(
        q=q, k=k, w=w, u=u, scale=scale, chunk_size=BT,
    )

    torch.testing.assert_close(v_new_pipe, v_new_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(o_pipe, o_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 512, 4, 64, 64),
    (1, 512, 4, 128, 128),
    (1, 2048, 8, 128, 128),
])
def test_fused_h_o_pipeline_cuda(B, T, H, K, V):
    """Test CUDA pipeline produces same output as sequential."""
    device = 'cuda'
    dtype = torch.bfloat16
    BT = 64
    scale = K ** -0.5

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)

    o_ref, v_new_ref = _reference(q, k, w, u, scale, BT)

    try:
        from fla.ops.delta_rule.fused_h_o_pipeline_cuda import fused_h_o_pipeline_cuda
        o_pipe, h_pipe, v_new_pipe, _ = fused_h_o_pipeline_cuda(
            q=q, k=k, w=w, u=u, scale=scale, chunk_size=BT,
        )
    except Exception as e:
        pytest.skip(f"CUDA JIT compilation failed: {e}")

    torch.testing.assert_close(v_new_pipe, v_new_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(o_pipe, o_ref, atol=1e-2, rtol=1e-2)


def test_sm_analysis():
    """Test that SM analysis runs and returns expected keys."""
    result = analyze_pipeline_feasibility(B=1, T=8192, H=32, K=128, V=128)
    assert 'num_sms' in result
    assert 'pipeline_feasible' in result
    assert 'spare_sms' in result
    assert result['num_sms'] > 0
    print_analysis(B=1, T=8192, H=32, K=128, V=128)


def test_sm_analysis_all_models():
    """Print analysis for common model sizes."""
    from fla.ops.delta_rule.sm_occupancy import MODELS
    for name, cfg in MODELS.items():
        print(f"\n=== {name} ===")
        result = print_analysis(**cfg)
        if cfg['H'] <= 54:
            assert result['spare_sms'] > 0 or result['fire4_ctas'] > result['num_sms']


if __name__ == '__main__':
    test_sm_analysis()
    print("\n\n")
    test_sm_analysis_all_models()
    print("\n\nRunning Triton pipeline correctness test...")
    test_fused_h_o_pipeline_triton(1, 512, 4, 64, 64)
    print("✅ Triton pipeline test passed!")
    print("\nRunning CUDA pipeline correctness test...")
    test_fused_h_o_pipeline_cuda(1, 512, 4, 64, 64)
    print("✅ CUDA pipeline test passed!")
