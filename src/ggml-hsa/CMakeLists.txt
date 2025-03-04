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

option(GGML_HSA_CPU_FALLBACK "ggml: use fallback to CPU if HSA kernel not implemented" OFF)

set(GGML_HEADERS_HSA ../../include/ggml-hsa.h)
set(GGML_SOURCES_HSA
    add.cpp
    cpy.cpp
    ggml-hsa.cpp
    kernel_registry.cpp
    mul_mat.cpp
    )

ggml_add_backend_library(ggml-hsa
                         ${GGML_HEADERS_HSA}
                         ${GGML_SOURCES_HSA}
                         )

#
# Kernel generation
#

find_package(PythonInterp REQUIRED)
find_package(Python REQUIRED)

function(add_aie_kernel TARGET KERNEL DEV DTYPE DIMS)
    set(KERNEL_NAME "${KERNEL}-${DEV}-${DTYPE}-${DIMS}")
    set(KERNEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KERNEL}.py")
    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernels")

    set(MLIR_OUTPUT "${OUTPUT_DIR}/${KERNEL_NAME}.mlir")
    add_custom_command(
        OUTPUT ${MLIR_OUTPUT}
        COMMAND ${PYTHON_EXECUTABLE} ${KERNEL_FILE} --dev ${DEV} --dtype ${DTYPE} --dims ${DIMS} > ${MLIR_OUTPUT}
        DEPENDS ${KERNEL_FILE}
        COMMENT "Generating AIE MLIR file for ${KERNEL_NAME} in ${MLIR_OUTPUT}"
        VERBATIM
    )

    set(XCLBIN_OUTPUT "${OUTPUT_DIR}/${KERNEL_NAME}.xclbin")
    set(INSTS_OUTPUT "${OUTPUT_DIR}/${KERNEL_NAME}_insts.txt")
    add_custom_command(
        OUTPUT ${XCLBIN_OUTPUT}
        COMMAND aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host --no-xchesscc --no-xbridge
                --xclbin-name=${XCLBIN_OUTPUT} --npu-insts-name=${INSTS_OUTPUT} ${MLIR_OUTPUT}
        DEPENDS ${MLIR_OUTPUT}
        COMMENT "Generating xclbin for ${KERNEL_NAME} in ${XCLBIN_OUTPUT}"
        VERBATIM
    )

    set(AIE_PARTITION_OUTPUT "${OUTPUT_DIR}/${KERNEL_NAME}_partition.json")
    add_custom_command(
        OUTPUT ${AIE_PARTITION_OUTPUT}
        COMMAND xclbinutil --dump-section AIE_PARTITION:JSON:${AIE_PARTITION_OUTPUT} --force --input ${XCLBIN_OUTPUT}
        DEPENDS ${XCLBIN_OUTPUT}
        COMMENT "Extracting PDI for ${KERNEL_NAME} with JSON in ${AIE_PARTITION_OUTPUT}"
        VERBATIM
    )

    set(PDI_OUTPUT "${OUTPUT_DIR}/${KERNEL_NAME}.pdi")
    add_custom_command(
        OUTPUT ${PDI_OUTPUT}
        COMMAND bash -c "cp ${OUTPUT_DIR}/`jq --raw-output \".aie_partition.PDIs.[0].file_name\" ${AIE_PARTITION_OUTPUT}` ${PDI_OUTPUT}"
        DEPENDS ${AIE_PARTITION_OUTPUT}
        COMMENT "Copying PDI for ${KERNEL_NAME} to ${PDI_OUTPUT}"
        VERBATIM
    )

    add_custom_target(ggml_hsa_kernel_${KERNEL_NAME} ALL DEPENDS ${PDI_OUTPUT})
    add_dependencies(${TARGET} ggml_hsa_kernel_${KERNEL_NAME})
endfunction()

add_aie_kernel(ggml-hsa add aie2 i32 16)
add_aie_kernel(ggml-hsa add aie2 i32 256)
add_aie_kernel(ggml-hsa add aie2 i32 512)

#
# Eigen
#

find_package(Eigen3 3.3 REQUIRED)

target_link_libraries(ggml-hsa PRIVATE Eigen3::Eigen)

#
#
#

target_link_libraries(ggml-hsa PRIVATE ggml-base hsa-runtime64::hsa-runtime64)
target_include_directories(ggml-hsa PRIVATE . ..)

target_sources(ggml-hsa PRIVATE common.hpp kernels.hpp)

set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)

if (GGML_HSA_CPU_FALLBACK)
    target_compile_definitions(ggml-hsa PRIVATE -DGGML_HSA_CPU_FALLBACK)
endif ()

add_compile_definitions(GGML_USE_HSA)
