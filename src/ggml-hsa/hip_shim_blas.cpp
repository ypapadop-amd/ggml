// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// hipBLAS shim: implements the GEMM entry points ggml-cuda calls (dense/batched
// attention & F16/F32 matmul) on top of a host BLAS (sgemm_) over UNIFIED memory.
// Correctness-first (Phase 4): the device pointers are CPU-accessible on the APU, so
// we drain the HSA queue and run the GEMM on the CPU. hipBLAS is column-major, matching
// the reference Fortran BLAS, so the layout maps 1:1. F16/BF16 operands are converted
// to F32 around the call. A native GPU GEMM can replace this later for performance.

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/library_types.h>   // hipDataType (HIP_R_16F/32F/16BF)
#include <hipblas/hipblas.h>     // hipblas* types + entry-point declarations

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Host reference/Fortran BLAS (column-major). Trailing hidden string-length args are
// not read by reference BLAS/OpenBLAS lsame(), so omitting them is safe here.
extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n,
                       const int * k, const float * alpha, const float * A, const int * lda,
                       const float * B, const int * ldb, const float * beta, float * C,
                       const int * ldc);

// Drain outstanding HSA work (defined in hip_shim.cpp) so operands are ready and results
// are visible before/after the CPU GEMM.
extern "C" hipError_t hipDeviceSynchronize();

namespace {

int elem_size(hipDataType t) {
    switch (t) {
        case HIP_R_32F:  return 4;
        case HIP_R_16F:  return 2;
        case HIP_R_16BF: return 2;
        default:         return 4;
    }
}

void to_f32(const void * src, hipDataType t, float * dst, size_t n) {
    if (t == HIP_R_32F) {
        std::memcpy(dst, src, n * sizeof(float));
    } else if (t == HIP_R_16F) {
        const _Float16 * s = static_cast<const _Float16 *>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(s[i]);
    } else if (t == HIP_R_16BF) {
        const uint16_t * s = static_cast<const uint16_t *>(src);
        for (size_t i = 0; i < n; ++i) {
            uint32_t u = static_cast<uint32_t>(s[i]) << 16;
            std::memcpy(&dst[i], &u, sizeof(float));
        }
    }
}

void from_f32(const float * src, void * dst, hipDataType t, size_t n) {
    if (t == HIP_R_32F) {
        std::memcpy(dst, src, n * sizeof(float));
    } else if (t == HIP_R_16F) {
        _Float16 * d = static_cast<_Float16 *>(dst);
        for (size_t i = 0; i < n; ++i) d[i] = static_cast<_Float16>(src[i]);
    } else if (t == HIP_R_16BF) {
        uint16_t * d = static_cast<uint16_t *>(dst);
        for (size_t i = 0; i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, &src[i], sizeof(float));
            d[i] = static_cast<uint16_t>(u >> 16);
        }
    }
}

char op_char(hipblasOperation_t op) { return (op == HIPBLAS_OP_N) ? 'N' : 'T'; }

// alpha/beta for the *Ex GEMMs are stored in the COMPUTE type (ggml uses F16 alpha/beta
// when compute_type==F16), so read them accordingly rather than assuming float.
float read_scalar(const void * p, hipblasComputeType_t ct, float dflt) {
    if (!p) return dflt;
    switch (ct) {
        case HIPBLAS_COMPUTE_16F: return static_cast<float>(*static_cast<const _Float16 *>(p));
        default:                  return *static_cast<const float *>(p); // 32F (also used for BF16 data)
    }
}

// C = alpha * op(A) * op(B) + beta * C, column-major, arbitrary element types (compute F32).
void gemm_one(hipblasOperation_t opA, hipblasOperation_t opB, int m, int n, int k, float alpha,
              const void * A, hipDataType tA, int lda, const void * B, hipDataType tB, int ldb,
              float beta, void * C, hipDataType tC, int ldc) {
    const size_t colsA = (opA == HIPBLAS_OP_N) ? (size_t) k : (size_t) m; // stored columns of A
    const size_t colsB = (opB == HIPBLAS_OP_N) ? (size_t) n : (size_t) k; // stored columns of B
    const size_t regionA = (size_t) lda * colsA;
    const size_t regionB = (size_t) ldb * colsB;
    const size_t regionC = (size_t) ldc * (size_t) n;

    std::vector<float> a32, b32, c32;
    const float * Ap;
    const float * Bp;
    float *       Cp;

    if (tA == HIP_R_32F) { Ap = static_cast<const float *>(A); }
    else { a32.resize(regionA); to_f32(A, tA, a32.data(), regionA); Ap = a32.data(); }

    if (tB == HIP_R_32F) { Bp = static_cast<const float *>(B); }
    else { b32.resize(regionB); to_f32(B, tB, b32.data(), regionB); Bp = b32.data(); }

    if (tC == HIP_R_32F) {
        Cp = static_cast<float *>(C);
    } else {
        c32.resize(regionC);
        if (beta != 0.0f) to_f32(C, tC, c32.data(), regionC); // preserve existing C for accumulation
        Cp = c32.data();
    }

    const char ta = op_char(opA);
    const char tb = op_char(opB);
    sgemm_(&ta, &tb, &m, &n, &k, &alpha, Ap, &lda, Bp, &ldb, &beta, Cp, &ldc);

    if (tC != HIP_R_32F) from_f32(Cp, C, tC, regionC);
}

} // namespace

extern "C" {

hipblasStatus_t hipblasCreate(hipblasHandle_t * handle) {
    if (handle) *handle = reinterpret_cast<hipblasHandle_t>(0x1);
    return HIPBLAS_STATUS_SUCCESS;
}
hipblasStatus_t hipblasDestroy(hipblasHandle_t) { return HIPBLAS_STATUS_SUCCESS; }
hipblasStatus_t hipblasSetStream(hipblasHandle_t, hipStream_t) { return HIPBLAS_STATUS_SUCCESS; }
hipblasStatus_t hipblasSetMathMode(hipblasHandle_t, hipblasMath_t) { return HIPBLAS_STATUS_SUCCESS; }

hipblasStatus_t hipblasSgemm(hipblasHandle_t, hipblasOperation_t transA, hipblasOperation_t transB,
                             int m, int n, int k, const float * alpha, const float * A, int lda,
                             const float * B, int ldb, const float * beta, float * C, int ldc) {
    (void) hipDeviceSynchronize();
    gemm_one(transA, transB, m, n, k, alpha ? *alpha : 1.0f, A, HIP_R_32F, lda, B, HIP_R_32F, ldb,
             beta ? *beta : 0.0f, C, HIP_R_32F, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasGemmEx(hipblasHandle_t, hipblasOperation_t transA, hipblasOperation_t transB,
                              int m, int n, int k, const void * alpha, const void * A,
                              hipDataType aType, int lda, const void * B, hipDataType bType, int ldb,
                              const void * beta, void * C, hipDataType cType, int ldc,
                              hipblasComputeType_t computeType, hipblasGemmAlgo_t) {
    (void) hipDeviceSynchronize();
    const float al = read_scalar(alpha, computeType, 1.0f);
    const float be = read_scalar(beta, computeType, 0.0f);
    gemm_one(transA, transB, m, n, k, al, A, aType, lda, B, bType, ldb, be, C, cType, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t, hipblasOperation_t transA,
                                            hipblasOperation_t transB, int m, int n, int k,
                                            const void * alpha, const void * A, hipDataType aType,
                                            int lda, hipblasStride strideA, const void * B,
                                            hipDataType bType, int ldb, hipblasStride strideB,
                                            const void * beta, void * C, hipDataType cType, int ldc,
                                            hipblasStride strideC, int batchCount,
                                            hipblasComputeType_t computeType, hipblasGemmAlgo_t) {
    (void) hipDeviceSynchronize();
    const float al = read_scalar(alpha, computeType, 1.0f);
    const float be = read_scalar(beta, computeType, 0.0f);
    const int esA = elem_size(aType), esB = elem_size(bType), esC = elem_size(cType);
    for (int i = 0; i < batchCount; ++i) {
        const void * Ai = static_cast<const char *>(A) + (long long) i * strideA * esA;
        const void * Bi = static_cast<const char *>(B) + (long long) i * strideB * esB;
        void *       Ci = static_cast<char *>(C) + (long long) i * strideC * esC;
        gemm_one(transA, transB, m, n, k, al, Ai, aType, lda, Bi, bType, ldb, be, Ci, cType, ldc);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t, hipblasOperation_t transA,
                                     hipblasOperation_t transB, int m, int n, int k,
                                     const void * alpha, const void * A[], hipDataType aType,
                                     int lda, const void * B[], hipDataType bType, int ldb,
                                     const void * beta, void * C[], hipDataType cType, int ldc,
                                     int batchCount, hipblasComputeType_t computeType,
                                     hipblasGemmAlgo_t) {
    (void) hipDeviceSynchronize();
    const float al = read_scalar(alpha, computeType, 1.0f);
    const float be = read_scalar(beta, computeType, 0.0f);
    for (int i = 0; i < batchCount; ++i) {
        gemm_one(transA, transB, m, n, k, al, A[i], aType, lda, B[i], bType, ldb, be, C[i], cType,
                 ldc);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

// F32 batched variants (used by OUT_PROD; not on the Llama path but cheap to support).
hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t, hipblasOperation_t transA,
                                           hipblasOperation_t transB, int m, int n, int k,
                                           const float * alpha, const float * A, int lda,
                                           long long strideA, const float * B, int ldb,
                                           long long strideB, const float * beta, float * C, int ldc,
                                           long long strideC, int batchCount) {
    (void) hipDeviceSynchronize();
    const float al = alpha ? *alpha : 1.0f, be = beta ? *beta : 0.0f;
    for (int i = 0; i < batchCount; ++i) {
        gemm_one(transA, transB, m, n, k, al, A + i * strideA, HIP_R_32F, lda, B + i * strideB,
                 HIP_R_32F, ldb, be, C + i * strideC, HIP_R_32F, ldc);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t, hipblasOperation_t transA,
                                    hipblasOperation_t transB, int m, int n, int k,
                                    const float * alpha, const float * const A[], int lda,
                                    const float * const B[], int ldb, const float * beta,
                                    float * const C[], int ldc, int batchCount) {
    (void) hipDeviceSynchronize();
    const float al = alpha ? *alpha : 1.0f, be = beta ? *beta : 0.0f;
    for (int i = 0; i < batchCount; ++i) {
        gemm_one(transA, transB, m, n, k, al, A[i], HIP_R_32F, lda, B[i], HIP_R_32F, ldb, be, C[i],
                 HIP_R_32F, ldc);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

// SOLVE_TRI only; not used by Llama. Fail loudly if ever reached.
hipblasStatus_t hipblasStrsmBatched(hipblasHandle_t, hipblasSideMode_t, hipblasFillMode_t,
                                    hipblasOperation_t, hipblasDiagType_t, int, int, const float *,
                                    const float * const[], int, float * const[], int, int) {
    fprintf(stderr, "[hip-shim] hipblasStrsmBatched not implemented (SOLVE_TRI unsupported)\n");
    abort();
}

} // extern "C"
