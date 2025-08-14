# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

include(ggml_hsa_utils)

if (GGML_HSA_JIT_COMPILE)
    # copy root package files
    set(IRON_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_files
        FILES
            ${IRON_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_files)
    install(
        FILES
            ${IRON_FILES}
        DESTINATION
            lib/iron
        )

    # copy utils package files
    set(IRON_UTILS_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/build.py
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/core_function.py
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/tensor_desc.py
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_utils_files
        FILES
            ${IRON_UTILS_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}/utils
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_utils_files)
    install(
        FILES
            ${IRON_UTILS_FILES}
        DESTINATION
            lib/iron/utils
        )

    # copy kernel files
    set(IRON_KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie_kernel_utils.h
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_kernel_files
        FILES
            ${IRON_KERNEL_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}/kernels
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_kernel_files)
    install(
        FILES
            ${IRON_KERNEL_FILES}
        DESTINATION
            lib/iron/kernels
        )

    # copy aie2 kernel files
    set(IRON_AIE2_KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie2/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie2/binary_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie2/mm.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie2/mul_mat.py
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie2/unary_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie2/unary_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aie2/zero.cc
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_aie2_kernel_files
        FILES
            ${IRON_AIE2_KERNEL_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}/kernels/aie2
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_aie2_kernel_files)
    install(
        FILES
            ${IRON_AIE2_KERNEL_FILES}
        DESTINATION
            lib/iron/kernels/aie2
        )
endif ()
