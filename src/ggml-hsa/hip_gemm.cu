// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// GPU GEMM for the hipBLAS shim (hip_shim_blas.cpp). This is a hipcc-compiled HIP TU: its <<<>>>
// launch funnels through the HIP host ABI (hip-abi.cpp) onto the native ggml-hsa dispatch, exactly
// like the ported ggml-cuda kernels. It replaces the correctness-first CPU sgemm_ that used to back
// the hipBLAS entry points, so f32/f16 dense + batched matmul now run on the gfx1151 iGPU.
//
// Convention matches cuBLAS/hipBLAS: column-major, C = alpha*op(A)*op(B) + beta*C. Operands may be
// F32/F16/BF16; the accumulation is always F32 and F16/BF16 operands are read + converted IN-KERNEL
// (no host round-trip). A classic shared-memory tiled kernel gives good occupancy on wave32 gfx1151
// (no MFMA); the tile size is a plain compile-time constant so it is trivially retunable.

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/library_types.h> // hipDataType (HIP_R_16F/32F/16BF)

#include <cstdint>
#include <cstdio>

namespace {

// ------------------------------------------------------------------------------------------------
// Element conversion helpers (device). BF16 is carried as its raw 16-bit storage; F16 uses __half.
// ------------------------------------------------------------------------------------------------
struct bf16_t {
    std::uint16_t x;
};

__device__ __forceinline__ float bf2f(std::uint16_t h) {
    union {
        std::uint32_t u;
        float         f;
    } c;
    c.u = static_cast<std::uint32_t>(h) << 16;
    return c.f;
}

__device__ __forceinline__ std::uint16_t f2bf(float f) {
    union {
        std::uint32_t u;
        float         f;
    } c;
    c.f = f;
    // round-to-nearest-even
    std::uint32_t lsb = (c.u >> 16) & 1u;
    c.u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(c.u >> 16);
}

__device__ __forceinline__ float cvt(float x) {
    return x;
}
__device__ __forceinline__ float cvt(__half x) {
    return __half2float(x);
}
__device__ __forceinline__ float cvt(bf16_t x) {
    return bf2f(x.x);
}

__device__ __forceinline__ void setv(float & d, float v) {
    d = v;
}
__device__ __forceinline__ void setv(__half & d, float v) {
    d = __float2half(v);
}
__device__ __forceinline__ void setv(bf16_t & d, float v) {
    d.x = f2bf(v);
}

// ------------------------------------------------------------------------------------------------
// Tiled column-major GEMM. op(A) is m x k, op(B) is k x n, C is m x n (all column-major).
//   opN == 0 -> no transpose (HIPBLAS_OP_N), opN == 1 -> transpose (HIPBLAS_OP_T).
// One thread computes one C[row,col]; a TS x TS shared tile marches along K.
// ------------------------------------------------------------------------------------------------
constexpr int TS = 16;

template <typename TA, typename TB, typename TC>
__global__ void gemm_kernel(int m, int n, int k, float alpha, const TA * __restrict__ A, int lda,
                            int opA, const TB * __restrict__ B, int ldb, int opB, float beta,
                            TC * __restrict__ C, int ldc) {
    __shared__ float As[TS][TS]; // As[ty][tx] = op(A)[row, kA]
    __shared__ float Bs[TS][TS]; // Bs[ty][tx] = op(B)[kB, col]

    const int row = blockIdx.y * TS + threadIdx.y; // i in [0,m)
    const int col = blockIdx.x * TS + threadIdx.x; // j in [0,n)

    float acc = 0.0f;

    const int ntiles = (k + TS - 1) / TS;
    for (int t = 0; t < ntiles; ++t) {
        const int kA = t * TS + threadIdx.x; // l-index for this thread's A load
        const int kB = t * TS + threadIdx.y; // l-index for this thread's B load

        if (row < m && kA < k) {
            // op(A)[row,kA]: N -> A[row + kA*lda]; T -> A[kA + row*lda]
            const long long idx = opA == 0 ? (long long) row + (long long) kA * lda
                                           : (long long) kA + (long long) row * lda;
            As[threadIdx.y][threadIdx.x] = cvt(A[idx]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && kB < k) {
            // op(B)[kB,col]: N -> B[kB + col*ldb]; T -> B[col + kB*ldb]
            const long long idx = opB == 0 ? (long long) kB + (long long) col * ldb
                                           : (long long) col + (long long) kB * ldb;
            Bs[threadIdx.y][threadIdx.x] = cvt(B[idx]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int e = 0; e < TS; ++e) {
            acc += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        const long long cidx = (long long) row + (long long) col * ldc;
        float out = alpha * acc;
        if (beta != 0.0f) {
            out += beta * cvt(C[cidx]);
        }
        setv(C[cidx], out);
    }
}

template <typename TA, typename TB, typename TC>
void launch(int opA, int opB, int m, int n, int k, float alpha, const void * A, int lda,
            const void * B, int ldb, float beta, void * C, int ldc) {
    const dim3 block(TS, TS, 1);
    const dim3 grid((n + TS - 1) / TS, (m + TS - 1) / TS, 1);
    gemm_kernel<TA, TB, TC><<<grid, block, 0, 0>>>(
        m, n, k, alpha, static_cast<const TA *>(A), lda, opA, static_cast<const TB *>(B), ldb, opB,
        beta, static_cast<TC *>(C), ldc);
}

// Dispatch TC given fixed TA,TB.
template <typename TA, typename TB>
void dispatch_C(int tC, int opA, int opB, int m, int n, int k, float alpha, const void * A, int lda,
                const void * B, int ldb, float beta, void * C, int ldc) {
    switch (tC) {
        case HIP_R_32F:
            launch<TA, TB, float>(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        case HIP_R_16F:
            launch<TA, TB, __half>(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        case HIP_R_16BF:
            launch<TA, TB, bf16_t>(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        default:
            fprintf(stderr, "[hip-gemm] unsupported C type %d\n", tC);
            break;
    }
}

// Dispatch TB given fixed TA.
template <typename TA>
void dispatch_B(int tB, int tC, int opA, int opB, int m, int n, int k, float alpha, const void * A,
                int lda, const void * B, int ldb, float beta, void * C, int ldc) {
    switch (tB) {
        case HIP_R_32F:
            dispatch_C<TA, float>(tC, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        case HIP_R_16F:
            dispatch_C<TA, __half>(tC, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        case HIP_R_16BF:
            dispatch_C<TA, bf16_t>(tC, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        default:
            fprintf(stderr, "[hip-gemm] unsupported B type %d\n", tB);
            break;
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Entry point called by hip_shim_blas.cpp (plain C++ TU). Types are hipDataType int codes; ops are
// 0 (N) / 1 (T). Column-major, C = alpha*op(A)*op(B) + beta*C. The launch is synchronous: hip-abi's
// hipLaunchKernel drains the native queue before returning, so C is visible on return.
// ------------------------------------------------------------------------------------------------
extern "C" void ggml_hsa_hip_gemm(int opA, int opB, int m, int n, int k, float alpha, const void * A,
                                  int tA, int lda, const void * B, int tB, int ldb, float beta,
                                  void * C, int tC, int ldc) {
    if (m <= 0 || n <= 0) {
        return;
    }
    switch (tA) {
        case HIP_R_32F:
            dispatch_B<float>(tB, tC, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        case HIP_R_16F:
            dispatch_B<__half>(tB, tC, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        case HIP_R_16BF:
            dispatch_B<bf16_t>(tB, tC, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        default:
            fprintf(stderr, "[hip-gemm] unsupported A type %d\n", tA);
            break;
    }
}
