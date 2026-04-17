// Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
// Fused 🔥4+🔥5 Pipeline CUDA Kernel — Header
// Author: zwx119 (ByteDance)
//
// Producer-Consumer persistent kernel:
//   Producer CTAs → 🔥4 (chunk_h recurrence)
//   Consumer CTAs → 🔥5 (chunk_o output computation)
//   Cross-CTA sync via __threadfence() + atomicAdd/volatile load
//
// Target: K=128, V=128, BT=64 (DeltaNet 7B config)

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ─── Compile-time constants ───
// Tile sizes for producer (🔥4)
constexpr int BT = 64;           // chunk size
constexpr int BK_BLOCK = 64;     // K-dimension block for state (K split into K/64 blocks)

// Tile sizes for consumer (🔥5) matmul
constexpr int CONS_BK_TILE = 32; // K-tile for shared-mem matmul in consumer
constexpr int CONS_BV_TILE = 32; // V-tile in consumer (if V > BV)

// Thread block config
constexpr int BLOCK_SIZE = 128;  // threads per CTA (4 warps)
constexpr int WARP_SIZE = 32;

// ─── Utility ───

// bf16 → float
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// float → bf16
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) {
    return __float2bfloat16(x);
}

// Volatile load for spin-waiting (bypass L1 cache)
__device__ __forceinline__ int volatile_load(const int* addr) {
    int val;
    asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

// ─── Small matmul helpers using shared memory ───
//
// These compute C[M, N] += A[M, K_tile] * B[K_tile, N] for small tiles,
// with A and B already in shared memory.
//
// Thread mapping: 128 threads → (thread_row, thread_col)
//   Each thread computes a TM × TN sub-tile of C.

// Generic shared-memory tiled matmul accumulator.
// A_smem: [M, K_tile], row-major
// B_smem: [K_tile, N], row-major
// C_reg:  [M, N] in registers (distributed across threads), row-major
//
// Thread layout: blockDim.x threads map to a 2D grid over (M, N).
//   thread (tr, tc) computes C[tr*TM .. (tr+1)*TM-1, tc*TN .. (tc+1)*TN-1]
//
// Template params:
//   M, N:     output tile dimensions
//   K_TILE:   inner dimension
//   TM, TN:   per-thread sub-tile
//   THREADS:  total threads in block

template <int M, int N, int K_TILE, int TM, int TN>
__device__ __forceinline__ void smem_matmul_acc(
    const float* __restrict__ A_smem,  // [M, K_TILE]
    const float* __restrict__ B_smem,  // [K_TILE, N]
    float C_reg[TM][TN],              // accumulated output per thread
    int tid
) {
    // Thread grid: (M/TM) × (N/TN) threads needed
    constexpr int ROWS = M / TM;
    constexpr int COLS = N / TN;
    int tr = tid / COLS;  // which row-block
    int tc = tid % COLS;  // which col-block

    if (tr >= ROWS) return;  // excess threads do nothing

    // Compute C_reg[TM][TN] += A[tr*TM..., :] * B[:, tc*TN...]
    for (int k = 0; k < K_TILE; k++) {
        float a_vals[TM];
        float b_vals[TN];

        #pragma unroll
        for (int m = 0; m < TM; m++) {
            a_vals[m] = A_smem[(tr * TM + m) * K_TILE + k];
        }
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            b_vals[n] = B_smem[k * N + tc * TN + n];
        }
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                C_reg[m][n] += a_vals[m] * b_vals[n];
            }
        }
    }
}

// ─── Cooperative load: global → shared memory ───
// Loads a [ROWS, COLS] tile from global memory (bf16) into shared memory (fp32).
// Threads cooperatively load, converting bf16 → fp32.
template <int ROWS, int COLS>
__device__ __forceinline__ void cooperative_load_bf16_to_fp32(
    float* __restrict__ smem,                    // [ROWS * COLS]
    const __nv_bfloat16* __restrict__ gmem,      // global base
    int row_stride,                              // stride between rows in global mem (in elements)
    int num_valid_rows,                          // actual valid rows (for boundary)
    int num_valid_cols,                          // actual valid cols (for boundary)
    int tid, int num_threads
) {
    int total = ROWS * COLS;
    for (int idx = tid; idx < total; idx += num_threads) {
        int r = idx / COLS;
        int c = idx % COLS;
        float val = 0.0f;
        if (r < num_valid_rows && c < num_valid_cols) {
            val = to_float(gmem[r * row_stride + c]);
        }
        smem[idx] = val;
    }
}

// Load fp32 global → fp32 shared
template <int ROWS, int COLS>
__device__ __forceinline__ void cooperative_load_fp32(
    float* __restrict__ smem,
    const float* __restrict__ gmem,
    int row_stride,
    int num_valid_rows,
    int num_valid_cols,
    int tid, int num_threads
) {
    int total = ROWS * COLS;
    for (int idx = tid; idx < total; idx += num_threads) {
        int r = idx / COLS;
        int c = idx % COLS;
        float val = 0.0f;
        if (r < num_valid_rows && c < num_valid_cols) {
            val = gmem[r * row_stride + c];
        }
        smem[idx] = val;
    }
}

// Store fp32 registers → global bf16
template <int TM, int TN, int M, int N, int COLS_THREADS>
__device__ __forceinline__ void store_reg_to_global_bf16(
    __nv_bfloat16* __restrict__ gmem,
    int row_stride,
    const float C_reg[TM][TN],
    int tid,
    int num_valid_rows,
    int num_valid_cols
) {
    int tr = tid / COLS_THREADS;
    int tc = tid % COLS_THREADS;

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = tr * TM + m;
        if (row >= num_valid_rows) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int col = tc * TN + n;
            if (col < num_valid_cols) {
                gmem[row * row_stride + col] = to_bf16(C_reg[m][n]);
            }
        }
    }
}

// Store fp32 registers → global fp32
template <int TM, int TN, int M, int N, int COLS_THREADS>
__device__ __forceinline__ void store_reg_to_global_fp32(
    float* __restrict__ gmem,
    int row_stride,
    const float C_reg[TM][TN],
    int tid,
    int num_valid_rows,
    int num_valid_cols
) {
    int tr = tid / COLS_THREADS;
    int tc = tid % COLS_THREADS;

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = tr * TM + m;
        if (row >= num_valid_rows) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int col = tc * TN + n;
            if (col < num_valid_cols) {
                gmem[row * row_stride + col] = C_reg[m][n];
            }
        }
    }
}
