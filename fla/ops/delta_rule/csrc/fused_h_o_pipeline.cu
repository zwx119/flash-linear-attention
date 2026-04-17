// Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
// Fused 🔥4+🔥5 Pipeline CUDA Kernel
// Author: zwx119 (ByteDance)
//
// Persistent kernel: producer CTAs run 🔥4, consumer CTAs run 🔥5.
// Cross-CTA sync: __threadfence() + atomicAdd (producer) / volatile_load (consumer).
//
// Key advantages over Triton version:
//   1. Guaranteed __threadfence() semantics (not inline asm workaround)
//   2. Proper volatile/cg loads for spin-wait (no atomic_add(ptr, 0) hack)
//   3. __launch_bounds__ for register control
//   4. Separate __noinline__ __device__ functions → compiler can optimize independently
//
// Target config: K∈{64,128,256}, V∈{64,128,256}, BT=64, bf16 input, fp32 accumulation.

#include "fused_h_o_pipeline.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ─────────────────────────────────────────────────────────────────────
// Shared memory layout (union for producer/consumer, since branches are exclusive)
// ─────────────────────────────────────────────────────────────────────
//
// Producer needs:
//   - state h: [K, BV_PROD] fp32 = K * BV_PROD * 4 bytes  (kept in regs where possible)
//   - W tile: [BT, 64] fp32 = 64 * 64 * 4 = 16 KB
//   - U tile: [BT, BV] fp32 = 64 * BV * 4 bytes
//   - K tile: [64, BT] fp32 = 64 * 64 * 4 = 16 KB
//   - v_new tile: [BT, BV] fp32 (can reuse U tile space)
//
// Consumer needs:
//   - Q tile: [BT, BK] fp32
//   - K tile: [BK, BT] fp32
//   - h tile: [BK, BV] fp32
//   - v tile: [BT, BV] fp32
//   - A tile: [BT, BT] fp32 (computed in registers or smem)
//
// Max shared memory: ~48KB per CTA by default, up to 164KB with dynamic allocation.
// We use dynamic shared memory for flexibility.


// ═══════════════════════════════════════════════════════════════════════
//  PRODUCER: 🔥4 sequential recurrence
// ═══════════════════════════════════════════════════════════════════════

// Producer CTA: one (i_v, i_n, i_h) combination.
// Iterates over NT chunks, maintaining state in registers + shared memory.
//
// For K=128, BV_PROD=64:
//   State = [128, 64] fp32 = 32KB → must be in shared memory
//   (Too much for registers: 32K/128threads = 256 regs/thread, but max is ~255)
//
// Strategy: split K into 64-wide blocks, process each sequentially.
//   Per K-block: state [64, BV] in shared memory, loaded to registers for matmul.

__device__ __noinline__ void producer_fn(
    // Inputs
    const __nv_bfloat16* __restrict__ k_ptr,   // [B*T, H, K]
    const __nv_bfloat16* __restrict__ w_ptr,   // [B*T, H, K]
    const __nv_bfloat16* __restrict__ u_ptr,   // [B*T, H, V]
    // Outputs
    float*              __restrict__ h_ptr,    // [B, NT, H, K, V]
    __nv_bfloat16*      __restrict__ vn_ptr,   // [B*T, H, V]
    int*                __restrict__ sem_ptr,  // [NT, B*H] semaphore
    // Indexing
    int i_v,            // V-slice index
    int i_n,            // batch index
    int i_h,            // head index
    // Dimensions
    int T, int H, int Hq, int K, int V,
    int NT, int BV_PROD, int BH,
    // Shared memory
    float* smem
) {
    int tid = threadIdx.x;
    int BH = 1 * H;  // for non-varlen, N=B already factored into i_n

    // ── Pointer offsets ──
    int bos = i_n * T;
    int i_hq = i_h / (H / Hq);

    // k: [B*T, Hq, K], row stride = Hq*K
    const __nv_bfloat16* k_base = k_ptr + (int64_t)(bos * Hq + i_hq) * K;
    int k_row_stride = Hq * K;

    // w: [B*T, H, K], row stride = H*K
    const __nv_bfloat16* w_base = w_ptr + (int64_t)(bos * H + i_h) * K;
    int w_row_stride = H * K;

    // u: [B*T, H, V], row stride = H*V
    const __nv_bfloat16* u_base = u_ptr + (int64_t)(bos * H + i_h) * V;
    int u_row_stride = H * V;

    // h: [B, NT, H, K, V]
    float* h_base = h_ptr + (int64_t)(i_n * NT * H + i_h) * K * V;

    // vn: same layout as u
    __nv_bfloat16* vn_base = vn_ptr + (int64_t)(bos * H + i_h) * V;
    int vn_row_stride = H * V;

    // sem: [NT, B*H]  (B*H = total batch*head count for this launch)
    int i_bh = i_n * H + i_h;

    // ── Shared memory partitioning ──
    // state_smem: [K, BV_PROD] fp32 — persistent across chunks
    // work_smem: temporary space for matmul tiles (reused per operation)
    int state_size = K * BV_PROD;          // e.g., 128*64 = 8192 floats = 32KB
    float* state_smem = smem;              // [K, BV_PROD]
    float* work_smem  = smem + state_size; // remaining space for tiles

    // Initialize state to zero
    for (int idx = tid; idx < state_size; idx += BLOCK_SIZE) {
        state_smem[idx] = 0.0f;
    }
    __syncthreads();

    // ── Main recurrence loop ──
    for (int i_t = 0; i_t < NT; i_t++) {
        int chunk_offset = i_t * BT;
        int valid_rows = min(BT, T - chunk_offset);

        // ──── 1. Store h[t] = current state to HBM ────
        // h_base + i_t * H*K*V, layout [K, V], only the [K, i_v*BV_PROD : (i_v+1)*BV_PROD] slice
        {
            float* h_out = h_base + (int64_t)i_t * H * K * V + i_v * BV_PROD;
            // h layout: [K, V], row stride = V
            for (int idx = tid; idx < K * BV_PROD; idx += BLOCK_SIZE) {
                int kr = idx / BV_PROD;
                int vc = idx % BV_PROD;
                h_out[kr * V + vc] = state_smem[kr * BV_PROD + vc];
            }
        }
        __syncthreads();

        // ──── 2. Compute v_new[t] = u[t] - W[t] · state ────
        // v_new: [BT, BV_PROD], W: [BT, K], state: [K, BV_PROD]
        // Split K into 64-wide blocks, accumulate W·state
        //
        // work_smem layout for this step:
        //   w_tile: [BT, 64] fp32  = BT*64 floats
        //   result: [BT, BV_PROD] fp32 = BT*BV_PROD floats  (accumulated matmul result)

        float* result_smem = work_smem;                     // [BT, BV_PROD]
        float* w_tile_smem = work_smem + BT * BV_PROD;      // [BT, 64]

        // Zero result
        for (int idx = tid; idx < BT * BV_PROD; idx += BLOCK_SIZE) {
            result_smem[idx] = 0.0f;
        }
        __syncthreads();

        int num_k_blocks = (K + 63) / 64;
        for (int ik = 0; ik < num_k_blocks; ik++) {
            int k_start = ik * 64;
            int k_width = min(64, K - k_start);

            // Load W[chunk_t, k_start:k_start+64] into w_tile_smem [BT, 64]
            // W is bf16, convert to fp32
            const __nv_bfloat16* w_src = w_base + (int64_t)chunk_offset * w_row_stride + k_start;
            cooperative_load_bf16_to_fp32<BT, 64>(
                w_tile_smem, w_src, w_row_stride,
                valid_rows, k_width, tid, BLOCK_SIZE
            );
            __syncthreads();

            // state_smem[k_start:k_start+64, :] is at state_smem + k_start * BV_PROD
            // Accumulate: result += w_tile · state_block
            // w_tile: [BT, 64], state_block: [64, BV_PROD] → result: [BT, BV_PROD]
            //
            // Each thread computes a sub-tile of the result.
            // Thread layout: (BT/TM) × (BV_PROD/TN) = e.g., (64/4) × (64/8) = 16 × 8 = 128
            constexpr int TM = 4;
            constexpr int TN = 8;
            constexpr int ROWS_T = BT / TM;    // 16
            constexpr int COLS_T = 8;           // BV_PROD / TN, but BV_PROD varies...

            // Simple loop-based approach for flexibility:
            float* state_block = state_smem + k_start * BV_PROD;
            int total_out = BT * BV_PROD;
            for (int idx = tid; idx < total_out; idx += BLOCK_SIZE) {
                int r = idx / BV_PROD;
                int c = idx % BV_PROD;
                float acc = 0.0f;
                for (int kk = 0; kk < k_width; kk++) {
                    acc += w_tile_smem[r * 64 + kk] * state_block[kk * BV_PROD + c];
                }
                result_smem[idx] += acc;
            }
            __syncthreads();
        }

        // result_smem now has W·state [BT, BV_PROD]
        // v_new = u[t] - result
        // Load u[t, i_v*BV_PROD : (i_v+1)*BV_PROD] and compute v_new, store to vn_base
        {
            const __nv_bfloat16* u_src = u_base + (int64_t)chunk_offset * u_row_stride + i_v * BV_PROD;
            __nv_bfloat16* vn_dst = vn_base + (int64_t)chunk_offset * vn_row_stride + i_v * BV_PROD;

            for (int idx = tid; idx < BT * BV_PROD; idx += BLOCK_SIZE) {
                int r = idx / BV_PROD;
                int c = idx % BV_PROD;
                float u_val = 0.0f;
                if (r < valid_rows) {
                    u_val = to_float(u_src[r * u_row_stride + c]);
                }
                float v_new_val = u_val - result_smem[idx];

                // Store to v_new global
                if (r < valid_rows) {
                    vn_dst[r * vn_row_stride + c] = to_bf16(v_new_val);
                }
                // Keep in result_smem for state update (reuse as v_new_smem)
                result_smem[idx] = v_new_val;
            }
        }
        __syncthreads();

        // ──── 3. __threadfence + signal consumer ────
        __threadfence();
        if (tid == 0) {
            atomicAdd(sem_ptr + i_t * BH + i_bh, 1);
        }

        // ──── 4. Update state: state += k^T · v_new ────
        // k[t]: [BT, K] (transposed as [K, BT] for matmul)
        // v_new: [BT, BV_PROD] (already in result_smem)
        // state += k^T · v_new = [K, BT] · [BT, BV_PROD] = [K, BV_PROD]

        // Process K in 64-wide blocks
        for (int ik = 0; ik < num_k_blocks; ik++) {
            int k_start = ik * 64;
            int k_width = min(64, K - k_start);

            // Load k[chunk_t, k_start:k_start+64] into w_tile_smem (reuse) as [BT, 64]
            const __nv_bfloat16* k_src = k_base + (int64_t)chunk_offset * k_row_stride + k_start;
            cooperative_load_bf16_to_fp32<BT, 64>(
                w_tile_smem, k_src, k_row_stride,
                valid_rows, k_width, tid, BLOCK_SIZE
            );
            __syncthreads();

            // Compute k^T · v_new = [64, BT] · [BT, BV_PROD] → [64, BV_PROD]
            // Accumulate into state_smem[k_start*BV_PROD ...]
            float* state_block = state_smem + k_start * BV_PROD;

            int total_out = k_width * BV_PROD;
            for (int idx = tid; idx < total_out; idx += BLOCK_SIZE) {
                int kr = idx / BV_PROD;
                int vc = idx % BV_PROD;
                float acc = 0.0f;
                for (int bt = 0; bt < valid_rows; bt++) {
                    // k^T[kr, bt] = k[bt, kr] = w_tile_smem[bt * 64 + kr]
                    acc += w_tile_smem[bt * 64 + kr] * result_smem[bt * BV_PROD + vc];
                }
                state_block[kr * BV_PROD + vc] += acc;
            }
            __syncthreads();
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  CONSUMER: 🔥5 per-chunk output computation
// ═══════════════════════════════════════════════════════════════════════

// Consumer CTA: one (i_v, i_t, i_b, i_h) work item.
// Waits for producer to finish chunk i_t, then computes:
//   o[t] = q[t] · h[t] * scale + tril(q[t] · k[t]^T) · v_new[t] * scale

__device__ __noinline__ void consumer_fn(
    // Inputs
    const __nv_bfloat16* __restrict__ q_ptr,   // [B*T, Hq, K]
    const __nv_bfloat16* __restrict__ k_ptr,   // [B*T, Hq, K]
    const float*         __restrict__ h_ptr,   // [B, NT, H, K, V]
    const __nv_bfloat16* __restrict__ vn_ptr,  // [B*T, H, V]
    // Output
    __nv_bfloat16*       __restrict__ o_ptr,   // [B*T, H, V]
    // Sync
    const int*           __restrict__ sem_ptr, // [NT * BH]
    // Indexing
    int i_v,           // output V-slice index
    int i_t,           // chunk index
    int i_b,           // batch index
    int i_h,           // head index
    // Dimensions
    int T, int H, int Hq, int K, int V,
    int NT, int BV_CONS, int BH,
    int num_producer_per_head,
    float scale,
    // Shared memory
    float* smem
) {
    int tid = threadIdx.x;
    int bos = i_b * T;
    int i_hq = i_h / (H / Hq);

    int chunk_offset = i_t * BT;
    int valid_rows = min(BT, T - chunk_offset);

    // ──── 1. Spin-wait for producer ────
    // sem[i_t * BH + i_bh] must reach num_producer_per_head
    int i_bh = i_b * H + i_h;
    const volatile int* sem_addr = (const volatile int*)(sem_ptr + i_t * BH + i_bh);

    if (tid == 0) {
        while (volatile_load((const int*)sem_addr) < num_producer_per_head) {
            // spin — only thread 0 polls to reduce atomic traffic
        }
    }
    __syncthreads();  // broadcast "ready" to all threads
    __threadfence();  // ensure we see producer's stores

    // ──── 2. Compute b_o = q · h [inter-chunk] and b_A = q · k^T [intra-chunk] ────
    // Shared memory layout:
    //   q_smem: [BT, BK_tile] fp32
    //   k_smem: [BK_tile, BT] fp32 (k transposed)
    //   h_smem: [BK_tile, BV_CONS] fp32
    //   (reused across K-tiles)

    int bk_tile = min(K, 64);  // inner tile size
    float* q_smem = smem;                                    // [BT, bk_tile]
    float* kt_smem = smem + BT * bk_tile;                   // [bk_tile, BT]
    float* h_smem = smem + BT * bk_tile + bk_tile * BT;     // [bk_tile, BV_CONS]
    // Total: BT*bk + bk*BT + bk*BV = bk*(2*BT + BV) = 64*(128+128) = 16K floats = 64KB (fits)

    // Accumulate o and A in registers (per-thread partial sums)
    // o: [BT, BV_CONS] → each thread owns a sub-tile
    // A: [BT, BT] → each thread owns a sub-tile
    //
    // For simplicity: accumulate in shared memory
    float* o_smem = smem;  // reuse at the end
    float* A_smem = smem + BT * BV_CONS;

    // Zero o_smem and A_smem
    for (int idx = tid; idx < BT * BV_CONS + BT * BT; idx += BLOCK_SIZE) {
        if (idx < BT * BV_CONS)
            o_smem[idx] = 0.0f;
        else
            A_smem[idx - BT * BV_CONS] = 0.0f;
    }
    __syncthreads();

    // Pointer setup
    const __nv_bfloat16* q_base = q_ptr + (int64_t)(bos * Hq + i_hq) * K;
    const __nv_bfloat16* k_base = k_ptr + (int64_t)(bos * Hq + i_hq) * K;
    const float* h_base = h_ptr + (int64_t)(i_b * NT * H + i_h) * K * V
                         + (int64_t)i_t * H * K * V;
    int q_row_stride = Hq * K;
    int k_row_stride = Hq * K;

    int num_k_tiles = (K + bk_tile - 1) / bk_tile;

    for (int ik = 0; ik < num_k_tiles; ik++) {
        int k_start = ik * bk_tile;
        int k_width = min(bk_tile, K - k_start);

        // Reload q_smem, kt_smem, h_smem for this tile
        float* q_tile = smem;
        float* kt_tile = smem + BT * bk_tile;
        float* h_tile = smem + BT * bk_tile + bk_tile * BT;

        // Load Q[t, k_start:k_start+bk] → q_tile [BT, bk_tile]
        {
            const __nv_bfloat16* q_src = q_base + (int64_t)chunk_offset * q_row_stride + k_start;
            cooperative_load_bf16_to_fp32<BT, 64>(
                q_tile, q_src, q_row_stride,
                valid_rows, k_width, tid, BLOCK_SIZE
            );
        }

        // Load K[t, k_start:k_start+bk]^T → kt_tile [bk_tile, BT]
        // k is [T, K] with row stride q_row_stride, we need transposed
        {
            const __nv_bfloat16* k_src = k_base + (int64_t)chunk_offset * k_row_stride + k_start;
            // Load as [BT, bk_tile] then transpose in smem
            float* k_temp = h_tile;  // temporarily reuse h_tile space
            cooperative_load_bf16_to_fp32<BT, 64>(
                k_temp, k_src, k_row_stride,
                valid_rows, k_width, tid, BLOCK_SIZE
            );
            __syncthreads();
            // Transpose: kt_tile[k, t] = k_temp[t, k]
            for (int idx = tid; idx < bk_tile * BT; idx += BLOCK_SIZE) {
                int kr = idx / BT;
                int tc = idx % BT;
                kt_tile[idx] = (kr < k_width && tc < valid_rows) ? k_temp[tc * 64 + kr] : 0.0f;
            }
        }

        // Load h[t, k_start:k_start+bk, i_v*BV:...] → h_tile [bk_tile, BV_CONS]
        // h layout: [K, V], row stride V
        {
            const float* h_src = h_base + k_start * V + i_v * BV_CONS;
            cooperative_load_fp32<64, 128>(
                h_tile, h_src, V,
                k_width, BV_CONS, tid, BLOCK_SIZE
            );
        }
        __syncthreads();

        // Accumulate: o_smem += q_tile · h_tile  ([BT, bk] · [bk, BV_CONS] → [BT, BV_CONS])
        for (int idx = tid; idx < BT * BV_CONS; idx += BLOCK_SIZE) {
            int r = idx / BV_CONS;
            int c = idx % BV_CONS;
            float acc = 0.0f;
            for (int kk = 0; kk < k_width; kk++) {
                acc += q_tile[r * bk_tile + kk] * h_tile[kk * BV_CONS + c];
            }
            o_smem[idx] += acc;
        }

        // Accumulate: A_smem += q_tile · kt_tile^T  but actually q·k^T
        // q_tile: [BT, bk], kt_tile: [bk, BT] → q_tile · kt_tile = [BT, BT]
        for (int idx = tid; idx < BT * BT; idx += BLOCK_SIZE) {
            int r = idx / BT;
            int c = idx % BT;
            float acc = 0.0f;
            for (int kk = 0; kk < k_width; kk++) {
                acc += q_tile[r * bk_tile + kk] * kt_tile[kk * BT + c];
            }
            A_smem[idx] += acc;
        }
        __syncthreads();
    }

    // ──── 3. Apply causal mask to A ────
    for (int idx = tid; idx < BT * BT; idx += BLOCK_SIZE) {
        int r = idx / BT;
        int c = idx % BT;
        int abs_r = chunk_offset + r;
        int abs_c = chunk_offset + c;
        // causal: keep if row >= col AND both in valid range
        if (!(abs_r >= abs_c && r < valid_rows && c < valid_rows)) {
            A_smem[idx] = 0.0f;
        }
    }
    __syncthreads();

    // ──── 4. Load v_new[t], compute o = o*scale + A·v_new*scale ────
    // v_new: [BT, BV_CONS] in HBM
    float* v_smem = smem + BT * BV_CONS + BT * BT;  // after o_smem and A_smem

    {
        const __nv_bfloat16* vn_src = vn_ptr + (int64_t)(bos * H + i_h) * V
                                     + (int64_t)chunk_offset * H * V + i_v * BV_CONS;
        int vn_row_stride = H * V;
        for (int idx = tid; idx < BT * BV_CONS; idx += BLOCK_SIZE) {
            int r = idx / BV_CONS;
            int c = idx % BV_CONS;
            v_smem[idx] = (r < valid_rows) ? to_float(vn_src[r * vn_row_stride + c]) : 0.0f;
        }
    }
    __syncthreads();

    // Compute A · v_new [BT, BT] · [BT, BV_CONS] → [BT, BV_CONS], add to o
    // Then multiply everything by scale and store
    {
        __nv_bfloat16* o_dst = o_ptr + (int64_t)(bos * H + i_h) * V
                              + (int64_t)chunk_offset * H * V + i_v * BV_CONS;
        int o_row_stride = H * V;

        for (int idx = tid; idx < BT * BV_CONS; idx += BLOCK_SIZE) {
            int r = idx / BV_CONS;
            int c = idx % BV_CONS;

            // A · v_new at [r, c]
            float av = 0.0f;
            for (int j = 0; j < BT; j++) {
                av += A_smem[r * BT + j] * v_smem[j * BV_CONS + c];
            }

            float o_val = o_smem[idx] * scale + av * scale;

            if (r < valid_rows) {
                o_dst[r * o_row_stride + c] = to_bf16(o_val);
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  MAIN PERSISTENT KERNEL
// ═══════════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
fused_h_o_pipeline_kernel_cuda(
    // Inputs
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ w_ptr,
    const __nv_bfloat16* __restrict__ u_ptr,
    // Outputs
    float*              __restrict__ h_ptr,
    __nv_bfloat16*      __restrict__ vn_ptr,
    __nv_bfloat16*      __restrict__ o_ptr,
    // Sync
    int*                __restrict__ sem_ptr,
    // Dims
    int T, int B, int H, int Hq, int K, int V,
    int NT, int BV_PROD, int BV_CONS,
    int num_producer_ctas,
    int num_producer_per_head,
    int BH,   // B * H (semaphore stride)
    float scale
) {
    extern __shared__ float smem[];

    int pid = blockIdx.x;

    if (pid < num_producer_ctas) {
        // ── Producer CTA ──
        int NH = B * H;
        int i_v  = pid / NH;
        int i_nh = pid % NH;
        int i_n  = i_nh / H;
        int i_h  = i_nh % H;

        producer_fn(
            k_ptr, w_ptr, u_ptr,
            h_ptr, vn_ptr, sem_ptr,
            i_v, i_n, i_h,
            T, H, Hq, K, V, NT, BV_PROD, BH,
            smem
        );
    } else {
        // ── Consumer CTA ──
        int consumer_id = pid - num_producer_ctas;
        int NV_cons = (V + BV_CONS - 1) / BV_CONS;
        int total_items = NV_cons * NT * B * H;

        if (consumer_id < total_items) {
            int i_v  = consumer_id / (NT * B * H);
            int rem  = consumer_id % (NT * B * H);
            int i_t  = rem / (B * H);
            int i_bh = rem % (B * H);
            int i_b  = i_bh / H;
            int i_h  = i_bh % H;

            consumer_fn(
                q_ptr, k_ptr, h_ptr, vn_ptr, o_ptr, sem_ptr,
                i_v, i_t, i_b, i_h,
                T, H, Hq, K, V, NT, BV_CONS, BH,
                num_producer_per_head, scale,
                smem
            );
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ LAUNCH WRAPPER
// ═══════════════════════════════════════════════════════════════════════

void fused_h_o_pipeline_launch(
    torch::Tensor q,       // [B, T, H, K] bf16
    torch::Tensor k,       // [B, T, H, K] bf16
    torch::Tensor w,       // [B, T, H, K] bf16
    torch::Tensor u,       // [B, T, H, V] bf16
    torch::Tensor h,       // [B, NT, H, K, V] fp32 (pre-allocated)
    torch::Tensor v_new,   // [B, T, H, V] bf16 (pre-allocated)
    torch::Tensor o,       // [B, T, H, V] bf16 (pre-allocated)
    torch::Tensor sem,     // [NT * B*H] int32, zero-initialized
    float scale,
    int BV_PROD,
    int BV_CONS
) {
    int B_val = q.size(0);
    int T_val = q.size(1);
    int Hq = q.size(2);
    int K_val = q.size(3);
    int H_val = w.size(2);
    int V_val = u.size(3);
    int BT_val = 64;
    int NT_val = (T_val + BT_val - 1) / BT_val;
    int BH = B_val * H_val;

    int num_producer_per_head = (V_val + BV_PROD - 1) / BV_PROD;
    int num_producer_ctas = num_producer_per_head * BH;

    int NV_cons = (V_val + BV_CONS - 1) / BV_CONS;
    int num_consumer_ctas = NV_cons * NT_val * BH;

    int total_ctas = num_producer_ctas + num_consumer_ctas;

    // Shared memory:
    // Producer needs: K*BV_PROD (state) + BT*BV_PROD (result) + BT*64 (tile) = at most ~48KB
    // Consumer needs: BT*BV_CONS + BT*BT + BT*64 + 64*BT + 64*BV_CONS ≈ varies
    int producer_smem = (K_val * BV_PROD + BT_val * BV_PROD + BT_val * 64) * sizeof(float);
    int consumer_smem = (BT_val * BV_CONS + BT_val * BT_val + BT_val * BV_CONS
                         + BT_val * 64 + 64 * BT_val + 64 * BV_CONS) * sizeof(float);
    int smem_bytes = max(producer_smem, consumer_smem);

    // Set max dynamic shared memory if needed
    auto stream = at::cuda::getCurrentCUDAStream();
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            fused_h_o_pipeline_kernel_cuda,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
    }

    fused_h_o_pipeline_kernel_cuda<<<total_ctas, BLOCK_SIZE, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(u.data_ptr()),
        h.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(v_new.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
        sem.data_ptr<int>(),
        T_val, B_val, H_val, Hq, K_val, V_val,
        NT_val, BV_PROD, BV_CONS,
        num_producer_ctas,
        num_producer_per_head,
        BH,
        scale
    );
}


// ═══════════════════════════════════════════════════════════════════════
//  PYBIND11 BINDING
// ═══════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_h_o_pipeline_launch", &fused_h_o_pipeline_launch,
          "Fused chunk_h + chunk_o pipeline kernel (CUDA)");
}
