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

option(GGML_HSA_JIT_COMPILE "ggml-hsa: enable JIT compilation of kernels" ON)

set(GGML_HEADERS_HSA ../../include/ggml-hsa.h)
set(GGML_SOURCES_HSA
    aie-kernel.cpp
    ggml-hsa.cpp
    host-ops.cpp
    kernel-discovery.cpp
    )

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
        aie-kernel.hpp
        common.hpp
        host-ops.hpp
        kernel-discovery.hpp
        type-traits.hpp
    )

if (GGML_HSA_JIT_COMPILE)
    target_sources(ggml-hsa PRIVATE kernel-compiler.hpp)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_JIT_COMPILE)
    target_link_libraries(ggml-hsa PRIVATE pybind11::embed)

    add_subdirectory(kernels)

    # Install kernel pregeneration tool and example configs
    install(
        PROGRAMS
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml-hsa-gen-kernels.py
        DESTINATION
            ${CMAKE_INSTALL_BINDIR}
        RENAME
            ggml-hsa-gen-kernels
    )
    install(
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/example-kernel-config.json
            ${CMAKE_CURRENT_SOURCE_DIR}/minimal-kernel-config.json
        DESTINATION
            share/ggml-hsa/examples
    )
endif ()

add_compile_definitions(GGML_USE_HSA)
