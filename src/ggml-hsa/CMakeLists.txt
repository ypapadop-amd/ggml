# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH "/usr")
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

# Forward AMDGPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
if (AMDGPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
endif()

find_package(hsa-runtime64 1.0 REQUIRED)

option(GGML_HSA_AIE         "ggml-hsa: enable the AIE (XDNA NPU) backend"  ON)
option(GGML_HSA_GPU         "ggml-hsa: enable the GPU (ROCm/HIP) backend"  OFF)
option(GGML_HSA_GPU_DELEGATE "ggml-hsa: run the ggml-cuda kernel set on the native HSA runtime (HIP host ABI)" OFF)
option(GGML_HSA_JIT_COMPILE "ggml-hsa: enable JIT compilation of kernels"  ON)

if (NOT GGML_HSA_AIE AND NOT GGML_HSA_GPU)
    message(FATAL_ERROR "ggml-hsa: at least one of GGML_HSA_AIE or GGML_HSA_GPU must be enabled")
endif ()

# JIT compilation (Python/pybind) is shared by the AIE (IRON/Triton -> pdi) and GPU
# (Triton -> hsaco) backends.

set(GGML_HEADERS_HSA ../../include/ggml-hsa.h)
set(GGML_SOURCES_HSA
    ggml-hsa.cpp
    host-ops.cpp
    kernel-discovery.cpp
    )

if (GGML_HSA_AIE)
    list(APPEND GGML_SOURCES_HSA aie-kernel.cpp)
endif ()

if (GGML_HSA_GPU)
    # gpu-kernel.cpp provides both the native (Triton) dispatch and the generic HIP dispatcher;
    # hsaco-metadata.cpp is the shared comgr code-object metadata parser.
    list(APPEND GGML_SOURCES_HSA gpu-kernel.cpp hsaco-metadata.cpp)
endif ()

if (GGML_HSA_GPU_DELEGATE)
    if (NOT GGML_HSA_GPU)
        message(FATAL_ERROR "GGML_HSA_GPU_DELEGATE requires GGML_HSA_GPU")
    endif ()

    # Compile the ggml-cuda kernel set with hipcc INTO the ggml-hsa library (no separate ggml-hip
    # target -> no circular dependency: hip-abi.cpp calls native ggml-hsa dispatch, and ggml-hsa
    # delegates to ggml-cuda). The HIP host runtime (~50 hip* symbols) is provided by hip-abi.cpp
    # on the native HSA machinery; hipBLAS by hip_shim_blas.cpp over a host BLAS (-fa off path).
    enable_language(HIP)

    file(GLOB   GGML_SOURCES_CUDA "../ggml-cuda/*.cu")
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    list(APPEND GGML_SOURCES_CUDA
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)

    list(APPEND GGML_SOURCES_HSA ${GGML_SOURCES_CUDA} hip-abi.cpp hip_shim_blas.cpp)
endif ()

if (GGML_HSA_JIT_COMPILE)
    include(FetchContent)

    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v3.0
        )

    set(PYBIND11_FINDPYTHON OFF)
    FetchContent_MakeAvailable(pybind11)

    set(GGML_SOURCES_HSA kernel-compiler.cpp ${GGML_SOURCES_HSA})
endif ()

ggml_add_backend_library(ggml-hsa
                         ${GGML_HEADERS_HSA}
                         ${GGML_SOURCES_HSA}
                         )

target_link_libraries(ggml-hsa PRIVATE ggml-base hsa-runtime64::hsa-runtime64)
target_include_directories(ggml-hsa PRIVATE . ..)

target_sources(ggml-hsa
    PRIVATE
        common.hpp
        host-ops.hpp
        kernel-discovery.hpp
        type-traits.hpp
    )

if (GGML_HSA_AIE)
    target_sources(ggml-hsa PRIVATE aie-kernel.hpp)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_AIE)
endif ()

if (GGML_HSA_GPU)
    # comgr (Code Object Manager) is used to read kernel argument metadata from .hsaco files.
    find_library(GGML_HSA_COMGR_LIB amd_comgr REQUIRED PATHS ${ROCM_PATH}/lib)
    target_sources(ggml-hsa PRIVATE gpu-kernel.hpp)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_GPU)
    target_include_directories(ggml-hsa PRIVATE ${ROCM_PATH}/include)
    target_link_libraries(ggml-hsa PRIVATE ${GGML_HSA_COMGR_LIB})
endif ()

if (GGML_HSA_GPU_DELEGATE)
    find_package(hip REQUIRED)

    # The ggml-cuda kernel sources are HIP device code; the HIP host ABI + hipBLAS shim + shared
    # metadata parser are plain C++ needing the HIP/HSA/comgr headers (not device code).
    set_source_files_properties(${GGML_SOURCES_CUDA} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(hip-abi.cpp hip_shim_blas.cpp hsaco-metadata.cpp PROPERTIES
        LANGUAGE CXX
        COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__")

    target_include_directories(ggml-hsa PRIVATE ../ggml-cuda ${ROCM_PATH}/include)
    target_compile_definitions(ggml-hsa PRIVATE
        GGML_HSA_GPU_DELEGATE GGML_USE_HIP GGML_HIP_NO_VMM GGML_CUDA_NO_VMM
        GGML_HIP_NO_MMQ_MFMA GGML_CUDA_FORCE_MMQ)
    # ggml-cuda.h / ggml-cuda registration are keyed off GGML_USE_CUDA.
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)

    # Fast math like CUDA's -use_fast_math (not -ffast-math: -ffinite-math-only breaks ggml's
    # INFINITY masking -> NaNs).
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -funsafe-math-optimizations")
    target_compile_options(ggml-hsa PRIVATE
        "$<$<COMPILE_LANGUAGE:HIP>:-ffast-math;-fno-finite-math-only>")

    # Host BLAS for the correctness-first attention/dense GEMM (hip_shim_blas.cpp, -fa off path).
    find_library(GGML_HSA_BLAS_LIB NAMES openblas blas)
    if (NOT GGML_HSA_BLAS_LIB)
        find_file(GGML_HSA_BLAS_LIB NAMES libopenblas.so.0 libblas.so.3
                  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu)
    endif ()
    if (NOT GGML_HSA_BLAS_LIB)
        message(FATAL_ERROR "GGML_HSA_GPU_DELEGATE: no BLAS found (install libopenblas-dev or libblas)")
    endif ()
    message(STATUS "ggml-hsa delegate BLAS: ${GGML_HSA_BLAS_LIB}")

    # hip::device provides the AMDGPU device libraries for the .cu compilation. Link with the C++
    # driver; the HIP language pulls libamdhip64 as an implicit NEEDED lib, so -Bsymbolic-functions
    # makes this library bind its own (hip-abi.cpp) hip* definitions first — the real HIP runtime is
    # never used.
    target_link_libraries(ggml-hsa PRIVATE hip::device ${GGML_HSA_BLAS_LIB})
    set_target_properties(ggml-hsa PROPERTIES LINKER_LANGUAGE CXX)
    target_link_options(ggml-hsa PRIVATE "-Wl,-Bsymbolic-functions")
endif ()

if (GGML_HSA_JIT_COMPILE)
    target_sources(ggml-hsa PRIVATE kernel-compiler.hpp)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_JIT_COMPILE)
    target_link_libraries(ggml-hsa PRIVATE pybind11::embed)

    add_subdirectory(kernels)
endif ()

add_compile_definitions(GGML_USE_HSA)
