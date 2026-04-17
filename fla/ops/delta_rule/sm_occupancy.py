# SM Occupancy Analyzer for DeltaNet 🔥4+🔥5 Pipeline Fusion
# Author: zwx119 (ByteDance)
#
# Determines whether the model configuration leaves enough spare SMs
# during 🔥4 (chunk_h) execution to overlap 🔥5 (chunk_o) work.

import torch
import triton


def get_gpu_sm_count(device: int | None = None) -> int:
    """Get the number of SMs on the current GPU."""
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count


def analyze_pipeline_feasibility(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    BT: int = 64,
    device: int | None = None,
) -> dict:
    """
    Analyze whether 🔥4+🔥5 pipeline fusion is beneficial.

    🔥4 grid = (cdiv(V, BV), N*H) where N=B for non-varlen, BV chosen by autotune.
    🔥5 grid = (cdiv(V, BV), NT, B*H) where NT = cdiv(T, BT).

    Pipeline is beneficial when 🔥4 uses fewer CTAs than available SMs,
    leaving spare SMs idle that could run 🔥5 consumer CTAs.

    Returns:
        dict with analysis results including:
        - num_sms: total SMs on GPU
        - fire4_ctas: number of 🔥4 CTAs
        - spare_sms: SMs not used by 🔥4
        - fire5_ctas_per_chunk: 🔥5 CTAs needed per chunk
        - pipeline_feasible: whether pipeline would help
        - expected_speedup_ms: rough estimate of time saved
    """
    num_sms = get_gpu_sm_count(device)
    NT = triton.cdiv(T, BT)

    # 🔥4 BV is autotuned; typical values: 32 or 64
    # On ada/hopper: BV ∈ {32, 64}; on ampere: BV = 32
    props = torch.cuda.get_device_properties(device or torch.cuda.current_device())
    cc = props.major * 10 + props.minor
    if cc >= 89:  # Ada / Hopper
        bv_h = 64
    else:
        bv_h = 32

    fire4_ctas = triton.cdiv(V, bv_h) * B * H

    # 🔥5 BV is autotuned; typical: 64 or 128 on hopper
    # Per-chunk CTAs = cdiv(V, BV_o) * B * H
    # But 🔥5 grid is (cdiv(V,BV), NT, B*H), so total = cdiv(V,BV)*NT*B*H
    # Per chunk: cdiv(V, BV_o) * B * H (all batch/head combos)
    # Actually for pipeline, consumer handles ONE head at a time (co-located with producer)
    # Per chunk per head: cdiv(V, BV_o)
    bv_o = 128 if cc >= 89 else 64
    fire5_ctas_per_chunk_per_head = triton.cdiv(V, bv_o)
    fire5_total_ctas = triton.cdiv(V, bv_o) * NT * B * H

    spare_sms = max(0, num_sms - fire4_ctas)
    # Can we fit at least 1 wave of 🔥5 per-chunk work in spare SMs?
    # Each producer CTA handles 1 (i_v, i_nh) slice across all NT chunks
    # So per chunk, fire4 produces cdiv(V, bv_h) * B * H tiles
    # Consumer needs cdiv(V, bv_o) * B * H CTAs per chunk
    fire5_ctas_per_chunk = fire5_ctas_per_chunk_per_head * B * H

    pipeline_feasible = spare_sms >= fire5_ctas_per_chunk

    # Rough timing estimate (from proposal profiling on 7B config)
    # 🔥4 ≈ 0.332ms, 🔥5 ≈ 0.298ms for B=1,T=8192,H=32,K=128,V=128
    # Scale linearly with NT for 🔥4, and with total CTAs for 🔥5
    fire4_time_est = 0.332 * (NT / 128) * (B * H / 32)
    fire5_time_est = 0.298 * (NT / 128) * (B * H / 32)

    if pipeline_feasible:
        # Pipeline: 🔥4 time + 1 chunk of 🔥5 latency
        pipeline_time = fire4_time_est + fire5_time_est / NT
        expected_speedup = fire5_time_est - fire5_time_est / NT
    else:
        pipeline_time = fire4_time_est + fire5_time_est
        expected_speedup = 0.0

    return {
        'num_sms': num_sms,
        'gpu_name': props.name,
        'compute_capability': f'{props.major}.{props.minor}',
        'fire4_ctas': fire4_ctas,
        'fire4_bv': bv_h,
        'fire5_ctas_per_chunk': fire5_ctas_per_chunk,
        'fire5_total_ctas': fire5_total_ctas,
        'fire5_bv': bv_o,
        'spare_sms': spare_sms,
        'NT': NT,
        'pipeline_feasible': pipeline_feasible,
        'fire4_time_est_ms': round(fire4_time_est, 3),
        'fire5_time_est_ms': round(fire5_time_est, 3),
        'pipeline_time_est_ms': round(pipeline_time, 3),
        'expected_speedup_ms': round(expected_speedup, 3),
    }


def print_analysis(
    B: int, T: int, H: int, K: int, V: int,
    BT: int = 64, device: int | None = None,
):
    """Pretty-print the SM occupancy analysis."""
    r = analyze_pipeline_feasibility(B, T, H, K, V, BT, device)
    print(f"{'='*60}")
    print(f"  🔥4+🔥5 Pipeline Feasibility Analysis")
    print(f"{'='*60}")
    print(f"  GPU: {r['gpu_name']} (SM {r['compute_capability']}, {r['num_sms']} SMs)")
    print(f"  Model: B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}")
    print(f"  NT (chunks): {r['NT']}")
    print(f"{'─'*60}")
    print(f"  🔥4 (chunk_h): {r['fire4_ctas']} CTAs (BV={r['fire4_bv']})")
    print(f"  🔥5 (chunk_o): {r['fire5_ctas_per_chunk']} CTAs/chunk (BV={r['fire5_bv']})")
    print(f"  Spare SMs:     {r['spare_sms']} = {r['num_sms']} - {r['fire4_ctas']}")
    print(f"{'─'*60}")
    if r['pipeline_feasible']:
        print(f"  ✅ Pipeline FEASIBLE")
        print(f"     {r['spare_sms']} spare SMs >= {r['fire5_ctas_per_chunk']} consumer CTAs/chunk")
        print(f"     Sequential: {r['fire4_time_est_ms']:.3f} + {r['fire5_time_est_ms']:.3f} "
              f"= {r['fire4_time_est_ms'] + r['fire5_time_est_ms']:.3f} ms")
        print(f"     Pipeline:   ~{r['pipeline_time_est_ms']:.3f} ms")
        print(f"     Speedup:    ~{r['expected_speedup_ms']:.3f} ms")
    else:
        print(f"  ❌ Pipeline NOT beneficial")
        print(f"     {r['spare_sms']} spare SMs < {r['fire5_ctas_per_chunk']} consumer CTAs/chunk")
        print(f"     🔥4 already saturates {r['fire4_ctas']}/{r['num_sms']} SMs")
    print(f"{'='*60}")
    return r


# ─── Common model configs ───
MODELS = {
    '1.3B': dict(B=1, T=8192, H=16, K=128, V=128),
    '2.7B': dict(B=1, T=8192, H=20, K=128, V=128),
    '7B':   dict(B=1, T=8192, H=32, K=128, V=128),
    '13B':  dict(B=1, T=8192, H=40, K=128, V=128),
    '30B':  dict(B=1, T=8192, H=56, K=128, V=128),
    '70B':  dict(B=1, T=8192, H=64, K=128, V=128),
}


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model = sys.argv[1]
        if model in MODELS:
            print_analysis(**MODELS[model])
        else:
            print(f"Unknown model: {model}. Available: {list(MODELS.keys())}")
    else:
        for name, cfg in MODELS.items():
            print(f"\n{'*'*60}")
            print(f"  Model: {name}")
            print_analysis(**cfg)
