#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

include(ggml_hsa_utils)

if (GGML_HSA_JIT_COMPILE)
    # copy support files
    set(IRON_SUPPORT_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/build.py
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/core_function.py
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/tensor_desc.py
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_support_files
        FILES
            ${IRON_SUPPORT_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}/utils
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_support_files)
    install(
        FILES
            ${IRON_SUPPORT_FILES}
        DESTINATION
            lib/iron/utils
        )

    # copy kernel sources
    set(IRON_AIE2_KERNEL_FILES
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
