#  Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

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

option(GGML_HSA_JIT_COMPILE "ggml-hsa: enable JIT compilation of kernels" ON)

option(GGML_HSA_CPU_FALLBACK "ggml-hsa: use fallback to CPU if HSA kernel not implemented" OFF)

set(GGML_HEADERS_HSA ../../include/ggml-hsa.h)
set(GGML_SOURCES_HSA
    ggml-hsa.cpp
    host-ops.cpp
    kernel-discovery.cpp
    )

if (GGML_HSA_JIT_COMPILE)
    include(FetchContent)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13
        )

    find_package(Python COMPONENTS Interpreter Development REQUIRED)
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
    )

if (GGML_HSA_JIT_COMPILE)
    add_subdirectory(iron_kernels)
    target_sources(ggml-hsa PRIVATE kernel-compiler.hpp)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_JIT_COMPILE)
    target_link_libraries(ggml-hsa PRIVATE pybind11::embed)
endif ()


if (GGML_HSA_CPU_FALLBACK)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_CPU_FALLBACK)
    target_link_libraries(ggml-hsa PRIVATE ${GGML_CPU_NAME})
endif ()

add_compile_definitions(GGML_USE_HSA)
