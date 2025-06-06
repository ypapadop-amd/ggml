#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

include(ggml_hsa_utils)

if (GGML_HSA_JIT_COMPILE)
    # copy support files
    set(IRON_SUPPORT_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/compiler.py
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_support_files
        FILES
            ${IRON_SUPPORT_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_support_files)
    install(
        FILES
            ${IRON_SUPPORT_FILES}
        DESTINATION
            lib/iron_kernels
        )

    # copy kernel sources
    set(IRON_KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/binary_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/mm.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mul_mat.py
        ${CMAKE_CURRENT_SOURCE_DIR}/unary_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/unary_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/zero.cc
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_kernel_files
        FILES
            ${IRON_KERNEL_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_kernel_files)
    install(
        FILES
            ${IRON_KERNEL_FILES}
        DESTINATION
            lib/iron_kernels
        )
endif ()
