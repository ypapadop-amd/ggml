# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

# copy root package files
set(TRITON_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd.py
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
    )
ggml_hsa_copy_files(ggml_hsa_copy_triton_files
    FILES
        ${TRITON_FILES}
    DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR}
    )
add_dependencies(ggml-hsa ggml_hsa_copy_triton_files)
install(
    FILES
        ${TRITON_FILES}
    DESTINATION
        lib/kernels/triton
    )
