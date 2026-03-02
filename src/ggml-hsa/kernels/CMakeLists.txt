# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

include(ggml_hsa_utils)

if (GGML_HSA_JIT_COMPILE)
    add_subdirectory(iron)

    # copy root package files
    set(JIT_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/binary_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/build.py
        ${CMAKE_CURRENT_SOURCE_DIR}/clamp.py
        ${CMAKE_CURRENT_SOURCE_DIR}/mul_mat.py
        ${CMAKE_CURRENT_SOURCE_DIR}/scale.py
        ${CMAKE_CURRENT_SOURCE_DIR}/soft_max.py
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_desc.py
        ${CMAKE_CURRENT_SOURCE_DIR}/unary_ops.py
        )
    ggml_hsa_copy_files(ggml_hsa_copy_jit_files
        FILES
            ${JIT_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}
        )
    add_dependencies(ggml-hsa ggml_hsa_copy_jit_files)
    install(
        FILES
            ${JIT_FILES}
        DESTINATION
            lib/kernels
        )
endif ()
