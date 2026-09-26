// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Shared scaffold for the ggml-hsa per-operator benchmarks: cross-backend
// selection (CPU / GPU / HSA, with a graceful skip when a backend is not
// compiled in) and a deterministic data filler. Each bench file includes this
// and keeps only its op-specific body and shape list.

#pragma once

#include <benchmark/benchmark.h>

#include <cstddef>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

enum class BackendType {
    CPU,
    GPU,
    HSA,
};

// Create the requested backend, or SkipWithError (returning nullptr) when that
// backend is not compiled in / fails to initialize. Callers only need to check
// for nullptr and return.
inline ggml_backend_t make_backend(BackendType type, benchmark::State & state) {
    ggml_backend_t backend = nullptr;
    switch (type) {
        case BackendType::CPU:
            backend = ggml_backend_cpu_init();
            break;
        case BackendType::GPU:
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
            break;
#else
            state.SkipWithError("CUDA backend not available.");
            return nullptr;
#endif
        case BackendType::HSA:
#ifdef GGML_USE_HSA
            backend = ggml_backend_hsa_init(0);
            break;
#else
            state.SkipWithError("HSA backend not available.");
            return nullptr;
#endif
    }
    if (backend == nullptr && !state.error_occurred()) {
        state.SkipWithError("Backend creation failed.");
    }
    return backend;
}

// Deterministic filler: v[i] = start + (i % 101). These ops are memory- or
// compute-bound, so the exact values do not affect the reported timings; pass a
// negative start when an op needs inputs that span zero.
inline std::vector<float> make_data(std::size_t n, float start = 0.0f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = start + static_cast<float>(i % 101);
    }
    return v;
}
