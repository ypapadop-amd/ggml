if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH "/usr")
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})

# Forward AMDGPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
if (AMDGPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
endif()

find_package(hsa-runtime64 1.0 REQUIRED)

message(STATUS "HSA found")

set(GGML_HEADERS_HSA ../../include/ggml-hsa.h)
set(GGML_SOURCES_HSA ggml-hsa.cpp)

ggml_add_backend_library(ggml-hsa
                         ${GGML_HEADERS_HSA}
                         ${GGML_SOURCES_HSA}
                         )

target_link_libraries(ggml-hsa PRIVATE ggml-base hsa-runtime64::hsa-runtime64)
target_include_directories(ggml-hsa PRIVATE . ..)

find_package(Eigen3 3.3 REQUIRED)
target_link_libraries(ggml-hsa PRIVATE Eigen3::Eigen)

target_sources(ggml-hsa PRIVATE common.hpp)

set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)

add_compile_definitions(GGML_USE_HSA)
