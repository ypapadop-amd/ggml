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

option(GGML_HSA_CPU_FALLBACK "ggml: use fallback to CPU if HSA kernel not implemented" OFF)

set(GGML_KERNEL_SOURCES_HSA
    add.cpp
    mul_mat.cpp
    )

set(GGML_HEADERS_HSA ../../include/ggml-hsa.h)
set(GGML_SOURCES_HSA
    ggml-hsa.cpp
    kernel_discovery.cpp
    ${GGML_KERNEL_SOURCES_HSA}
    )

ggml_add_backend_library(ggml-hsa
                         ${GGML_HEADERS_HSA}
                         ${GGML_SOURCES_HSA}
                         )

add_subdirectory(iron-kernels)

target_link_libraries(ggml-hsa PRIVATE ggml-base hsa-runtime64::hsa-runtime64)
target_include_directories(ggml-hsa PRIVATE . ..)

target_sources(ggml-hsa
    PRIVATE
        common.hpp kernel_discovery.hpp kernels.hpp
    )

set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)

if (GGML_HSA_CPU_FALLBACK)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_CPU_FALLBACK)
    target_link_libraries(ggml-hsa PRIVATE ggml-base ${GGML_CPU_NAME})
endif ()

add_compile_definitions(GGML_USE_HSA)
