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
// backend is not compiled into this build.
inline ggml_backend_t make_backend(BackendType type, benchmark::State & state) {
    switch (type) {
        case BackendType::CPU:
            return ggml_backend_cpu_init();
        case BackendType::GPU:
#ifdef GGML_USE_CUDA
            return ggml_backend_cuda_init(0);
#else
            state.SkipWithError("CUDA backend not available.");
            return nullptr;
#endif
        case BackendType::HSA:
#ifdef GGML_USE_HSA
            return ggml_backend_hsa_init(0);
#else
            state.SkipWithError("HSA backend not available.");
            return nullptr;
#endif
    }
    state.SkipWithError("Invalid backend type.");
    return nullptr;
}

// Deterministic filler: v[i] = start + step * (i % 101). These ops are memory-
// or compute-bound, so the exact values do not affect the reported timings.
inline std::vector<float> make_data(std::size_t n, float start = 0.0f, float step = 1.0f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = start + step * static_cast<float>(i % 101);
    }
    return v;
}
