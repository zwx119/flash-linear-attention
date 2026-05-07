// Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
// H100 warp-specialized DeltaNet triangular solve prototype.
//
// One CTA handles one (chunk, batch, head). Warp 0 computes the four diagonal
// 16x16 triangular inverses with scalar CUDA cores. Warps 1-3 wait on the
// diagonal readiness flags and compute off-diagonal 16x16 block products with
// WMMA bf16 tensor-core instructions. This is the first CUDA version that
// creates real intra-CTA overlap between the CC-heavy diagonal solve and the
// TC-heavy off-diagonal matmuls. It intentionally targets the benchmark path:
//   A:  [B, T, H, 64] fp32, contiguous, T % 64 == 0
//   Ai: [B, T, H, 64] bf16

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <stdint.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kBT = 64;
constexpr int kSub = 16;
constexpr int kThreads = 128;

enum OffBlock {
    kOff21 = 0,
    kOff31 = 1,
    kOff32 = 2,
    kOff41 = 3,
    kOff42 = 4,
    kOff43 = 5,
};

enum ReadyFlag {
    kReadyD0 = 0,
    kReadyD1 = 1,
    kReadyD2 = 2,
    kReadyD3 = 3,
    kReady21 = 4,
    kReady31 = 5,
    kReady32 = 6,
    kReady42 = 7,
    kReady43 = 8,
};

__device__ __forceinline__ int lane_id() {
    return threadIdx.x & 31;
}

__device__ __forceinline__ int warp_id() {
    return threadIdx.x >> 5;
}

__device__ __forceinline__ void wait_ready(volatile int* ready, int flag) {
    while (ready[flag] == 0) {
        // Keep the loop visible to the scheduler; flags are block-local.
        asm volatile("nop;");
    }
    __syncwarp();
}

__device__ __forceinline__ void set_ready(volatile int* ready, int flag) {
    __threadfence_block();
    if (lane_id() == 0) {
        ready[flag] = 1;
    }
    __syncwarp();
}

__device__ __forceinline__ __nv_bfloat16 bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float load_A(
    const float* __restrict__ A,
    int64_t base_row,
    int64_t row_stride,
    int row,
    int col
) {
    return A[(base_row + row) * row_stride + col];
}

__device__ void zero_output_chunk(
    __nv_bfloat16* __restrict__ Ai,
    int64_t base_row,
    int64_t row_stride
) {
    for (int idx = threadIdx.x; idx < kBT * kBT; idx += kThreads) {
        int row = idx / kBT;
        int col = idx % kBT;
        Ai[(base_row + row) * row_stride + col] = bf16(0.0f);
    }
}

__device__ void load_offdiag_blocks(
    const float* __restrict__ A,
    int64_t base_row,
    int64_t row_stride,
    __nv_bfloat16* __restrict__ off_a
) {
    constexpr int ro[6] = {16, 32, 32, 48, 48, 48};
    constexpr int co[6] = {0, 0, 16, 0, 16, 32};
    for (int idx = threadIdx.x; idx < 6 * kSub * kSub; idx += kThreads) {
        int block = idx / (kSub * kSub);
        int elem = idx - block * kSub * kSub;
        int r = elem / kSub;
        int c = elem % kSub;
        float x = load_A(A, base_row, row_stride, ro[block] + r, co[block] + c);
        off_a[idx] = bf16(x);
    }
}

__device__ void solve_diag16_warp(
    const float* __restrict__ A,
    int64_t base_row,
    int64_t row_stride,
    int block_id,
    float* __restrict__ diag_f,
    __nv_bfloat16* __restrict__ diag_b,
    float* __restrict__ row_tmp
) {
    int lane = lane_id();
    int row0 = block_id * kSub;
    int col0 = block_id * kSub;

    for (int idx = lane; idx < kSub * kSub; idx += 32) {
        int r = idx / kSub;
        int c = idx % kSub;
        float x = 0.0f;
        if (r > c) {
            x = -load_A(A, base_row, row_stride, row0 + r, col0 + c);
        }
        diag_f[idx] = x;
    }
    __syncwarp();

    // Match FLA solve_tril: diagonal identity is added after the recursive
    // lower-triangular rows are solved, not during the recurrence.
    for (int i = 2; i < kSub; ++i) {
        if (lane < kSub) {
            row_tmp[lane] = lane < i ? -load_A(A, base_row, row_stride, row0 + i, col0 + lane) : 0.0f;
        }
        __syncwarp();

        if (lane < kSub) {
            float acc = row_tmp[lane];
            #pragma unroll
            for (int k = 0; k < kSub; ++k) {
                acc += row_tmp[k] * diag_f[k * kSub + lane];
            }
            diag_f[i * kSub + lane] = acc;
        }
        __syncwarp();
    }

    for (int idx = lane; idx < kSub * kSub; idx += 32) {
        int r = idx / kSub;
        int c = idx % kSub;
        float x = diag_f[idx] + (r == c ? 1.0f : 0.0f);
        diag_f[idx] = x;
        diag_b[idx] = bf16(x);
    }
    __syncwarp();
}

__device__ __forceinline__ void mma_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ c
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);
    wmma::load_matrix_sync(frag_a, a, 16);
    wmma::load_matrix_sync(frag_b, b, 16);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    wmma::store_matrix_sync(c, frag_c, 16, wmma::mem_row_major);
}

__device__ void float_to_bf16_warp(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst
) {
    int lane = lane_id();
    for (int idx = lane; idx < kSub * kSub; idx += 32) {
        dst[idx] = bf16(src[idx]);
    }
    __syncwarp();
}

__device__ void negate_float_warp(float* __restrict__ x) {
    int lane = lane_id();
    for (int idx = lane; idx < kSub * kSub; idx += 32) {
        x[idx] = -x[idx];
    }
    __syncwarp();
}

__device__ void add_float_warp(
    float* __restrict__ dst,
    const float* __restrict__ src
) {
    int lane = lane_id();
    for (int idx = lane; idx < kSub * kSub; idx += 32) {
        dst[idx] += src[idx];
    }
    __syncwarp();
}

__device__ void two_mma_neg_warp(
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ mid,
    const __nv_bfloat16* __restrict__ right,
    float* __restrict__ out,
    __nv_bfloat16* __restrict__ out_b,
    float* __restrict__ tmp_f,
    __nv_bfloat16* __restrict__ tmp_b
) {
    mma_bf16(left, mid, tmp_f);
    float_to_bf16_warp(tmp_f, tmp_b);
    mma_bf16(tmp_b, right, out);
    negate_float_warp(out);
    float_to_bf16_warp(out, out_b);
}

__device__ void left_mma_sum_neg_warp(
    const __nv_bfloat16* __restrict__ left,
    float* __restrict__ sum_f,
    __nv_bfloat16* __restrict__ sum_b,
    float* __restrict__ out,
    __nv_bfloat16* __restrict__ out_b
) {
    float_to_bf16_warp(sum_f, sum_b);
    mma_bf16(left, sum_b, out);
    negate_float_warp(out);
    float_to_bf16_warp(out, out_b);
}

__device__ void store_bf16_block(
    __nv_bfloat16* __restrict__ Ai,
    int64_t base_row,
    int64_t row_stride,
    int row0,
    int col0,
    const __nv_bfloat16* __restrict__ block
) {
    for (int idx = threadIdx.x; idx < kSub * kSub; idx += kThreads) {
        int r = idx / kSub;
        int c = idx % kSub;
        Ai[(base_row + row0 + r) * row_stride + col0 + c] = block[idx];
    }
}

__global__ __launch_bounds__(kThreads, 2)
void overlap_solve_tril_64x64_kernel(
    const float* __restrict__ A,
    __nv_bfloat16* __restrict__ Ai,
    int T,
    int H
) {
    int i_t = blockIdx.x;
    int i_bh = blockIdx.y;

    int base_token = i_t * kBT;
    int64_t row_stride = static_cast<int64_t>(H) * kBT;
    int64_t base_row = static_cast<int64_t>(base_token) * H + i_bh;

    __shared__ __align__(16) float sh_diag_f[4][kSub * kSub];
    __shared__ __align__(16) __nv_bfloat16 sh_diag_b[4][kSub * kSub];
    __shared__ __align__(16) __nv_bfloat16 sh_a[6][kSub * kSub];
    __shared__ __align__(16) float sh_off_f[6][kSub * kSub];
    __shared__ __align__(16) __nv_bfloat16 sh_off_b[6][kSub * kSub];
    __shared__ __align__(16) float sh_tmp_f[4][2][kSub * kSub];
    __shared__ __align__(16) __nv_bfloat16 sh_tmp_b[4][kSub * kSub];
    __shared__ __align__(16) float sh_row_tmp[kSub];
    __shared__ volatile int ready[9];

    if (threadIdx.x < 9) {
        ready[threadIdx.x] = 0;
    }

    zero_output_chunk(Ai, base_row, row_stride);
    load_offdiag_blocks(A, base_row, row_stride, &sh_a[0][0]);
    __syncthreads();

    int wid = warp_id();

    if (wid == 0) {
        solve_diag16_warp(A, base_row, row_stride, 0, sh_diag_f[0], sh_diag_b[0], sh_row_tmp);
        set_ready(ready, kReadyD0);
        solve_diag16_warp(A, base_row, row_stride, 1, sh_diag_f[1], sh_diag_b[1], sh_row_tmp);
        set_ready(ready, kReadyD1);
        solve_diag16_warp(A, base_row, row_stride, 2, sh_diag_f[2], sh_diag_b[2], sh_row_tmp);
        set_ready(ready, kReadyD2);
        solve_diag16_warp(A, base_row, row_stride, 3, sh_diag_f[3], sh_diag_b[3], sh_row_tmp);
        set_ready(ready, kReadyD3);
    } else if (wid == 1) {
        // Ai21 = -I22 A21 I11
        wait_ready(ready, kReadyD0);
        wait_ready(ready, kReadyD1);
        two_mma_neg_warp(
            sh_diag_b[1], sh_a[kOff21], sh_diag_b[0],
            sh_off_f[kOff21], sh_off_b[kOff21],
            sh_tmp_f[wid][0], sh_tmp_b[wid]
        );
        set_ready(ready, kReady21);

        // Ai31 = -I33 (A31 I11 + A32 Ai21)
        wait_ready(ready, kReadyD2);
        wait_ready(ready, kReady21);
        mma_bf16(sh_a[kOff31], sh_diag_b[0], sh_tmp_f[wid][0]);
        mma_bf16(sh_a[kOff32], sh_off_b[kOff21], sh_tmp_f[wid][1]);
        add_float_warp(sh_tmp_f[wid][0], sh_tmp_f[wid][1]);
        left_mma_sum_neg_warp(
            sh_diag_b[2], sh_tmp_f[wid][0], sh_tmp_b[wid],
            sh_off_f[kOff31], sh_off_b[kOff31]
        );
        set_ready(ready, kReady31);

        // Ai41 = -I44 (A41 I11 + A42 Ai21 + A43 Ai31)
        wait_ready(ready, kReadyD3);
        wait_ready(ready, kReady31);
        mma_bf16(sh_a[kOff41], sh_diag_b[0], sh_tmp_f[wid][0]);
        mma_bf16(sh_a[kOff42], sh_off_b[kOff21], sh_tmp_f[wid][1]);
        add_float_warp(sh_tmp_f[wid][0], sh_tmp_f[wid][1]);
        mma_bf16(sh_a[kOff43], sh_off_b[kOff31], sh_tmp_f[wid][1]);
        add_float_warp(sh_tmp_f[wid][0], sh_tmp_f[wid][1]);
        left_mma_sum_neg_warp(
            sh_diag_b[3], sh_tmp_f[wid][0], sh_tmp_b[wid],
            sh_off_f[kOff41], sh_off_b[kOff41]
        );
    } else if (wid == 2) {
        // Ai32 = -I33 A32 I22
        wait_ready(ready, kReadyD1);
        wait_ready(ready, kReadyD2);
        two_mma_neg_warp(
            sh_diag_b[2], sh_a[kOff32], sh_diag_b[1],
            sh_off_f[kOff32], sh_off_b[kOff32],
            sh_tmp_f[wid][0], sh_tmp_b[wid]
        );
        set_ready(ready, kReady32);

        // Ai42 = -I44 (A42 I22 + A43 Ai32)
        wait_ready(ready, kReadyD3);
        wait_ready(ready, kReady32);
        mma_bf16(sh_a[kOff42], sh_diag_b[1], sh_tmp_f[wid][0]);
        mma_bf16(sh_a[kOff43], sh_off_b[kOff32], sh_tmp_f[wid][1]);
        add_float_warp(sh_tmp_f[wid][0], sh_tmp_f[wid][1]);
        left_mma_sum_neg_warp(
            sh_diag_b[3], sh_tmp_f[wid][0], sh_tmp_b[wid],
            sh_off_f[kOff42], sh_off_b[kOff42]
        );
        set_ready(ready, kReady42);
    } else if (wid == 3) {
        // Ai43 = -I44 A43 I33
        wait_ready(ready, kReadyD2);
        wait_ready(ready, kReadyD3);
        two_mma_neg_warp(
            sh_diag_b[3], sh_a[kOff43], sh_diag_b[2],
            sh_off_f[kOff43], sh_off_b[kOff43],
            sh_tmp_f[wid][0], sh_tmp_b[wid]
        );
        set_ready(ready, kReady43);
    }

    __syncthreads();

    store_bf16_block(Ai, base_row, row_stride, 0, 0, sh_diag_b[0]);
    store_bf16_block(Ai, base_row, row_stride, 16, 16, sh_diag_b[1]);
    store_bf16_block(Ai, base_row, row_stride, 32, 32, sh_diag_b[2]);
    store_bf16_block(Ai, base_row, row_stride, 48, 48, sh_diag_b[3]);
    store_bf16_block(Ai, base_row, row_stride, 16, 0, sh_off_b[kOff21]);
    store_bf16_block(Ai, base_row, row_stride, 32, 0, sh_off_b[kOff31]);
    store_bf16_block(Ai, base_row, row_stride, 32, 16, sh_off_b[kOff32]);
    store_bf16_block(Ai, base_row, row_stride, 48, 0, sh_off_b[kOff41]);
    store_bf16_block(Ai, base_row, row_stride, 48, 16, sh_off_b[kOff42]);
    store_bf16_block(Ai, base_row, row_stride, 48, 32, sh_off_b[kOff43]);
}

torch::Tensor overlap_solve_tril_fwd(torch::Tensor A) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be fp32");
    TORCH_CHECK(A.dim() == 4, "A must have shape [B, T, H, 64]");
    TORCH_CHECK(A.size(3) == kBT, "A.shape[-1] must be 64");
    TORCH_CHECK(A.size(1) % kBT == 0, "T must be divisible by 64");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");

    const int B = static_cast<int>(A.size(0));
    const int T = static_cast<int>(A.size(1));
    const int H = static_cast<int>(A.size(2));
    const int NT = T / kBT;

    auto Ai = torch::empty(
        {A.size(0), A.size(1), A.size(2), A.size(3)},
        A.options().dtype(torch::kBFloat16)
    );

    dim3 grid(NT, B * H);
    dim3 block(kThreads);
    auto stream = at::cuda::getCurrentCUDAStream();
    overlap_solve_tril_64x64_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(Ai.data_ptr<at::BFloat16>()),
        T,
        H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Ai;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("overlap_solve_tril_fwd", &overlap_solve_tril_fwd, "H100 overlap solve_tril forward");
}
