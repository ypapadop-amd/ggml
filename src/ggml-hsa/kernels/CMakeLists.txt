# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

include(ggml_hsa_utils)

if (GGML_HSA_JIT_COMPILE)
    # copy root package files
    set(IRON_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/aie_kernel_utils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/binary_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/build.py
        ${CMAKE_CURRENT_SOURCE_DIR}/gemm.py
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml-aie.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_desc.py
        ${CMAKE_CURRENT_SOURCE_DIR}/unary_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/unary_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/scale.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scale.py
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
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

    # copy aie2 kernel files
    set(IRON_AIE2_KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/aie2/mm.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aie2/zero.cc
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_aie2_kernel_files
        FILES
            ${IRON_AIE2_KERNEL_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}/aie2
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_aie2_kernel_files)
    install(
        FILES
            ${IRON_AIE2_KERNEL_FILES}
        DESTINATION
            lib/iron/aie2
        )

    # copy aie2p kernel files
    set(IRON_AIE2P_KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/aie2p/mm.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aie2p/zero.cc
        )
    ggml_hsa_copy_files(ggml_hsa_copy_iron_aie2p_kernel_files
        FILES
            ${IRON_AIE2P_KERNEL_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}/aie2p
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_iron_aie2p_kernel_files)
    install(
        FILES
            ${IRON_AIE2P_KERNEL_FILES}
        DESTINATION
            lib/iron/aie2p
        )
endif ()
