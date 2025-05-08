#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# copy compiler files to build and install directories

include(ggml_hsa_utils)

set(COMPILER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler.py
    )

ggml_hsa_copy_files(ggml_hsa_copy_compiler_files
    FILES
        ${COMPILER_FILES}
    DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR}
    )

add_dependencies(ggml-hsa ggml_hsa_copy_compiler_files)

install(
    FILES
        ${COMPILER_FILES}
    DESTINATION
        lib/iron_kernels/compiler
    )
