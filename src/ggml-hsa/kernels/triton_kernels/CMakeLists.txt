# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

# copy root package files
set(TRITON_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/argmax.py
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_ops.py
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_2d.py
    ${CMAKE_CURRENT_SOURCE_DIR}/count_equal.py
    ${CMAKE_CURRENT_SOURCE_DIR}/cross_entropy_loss.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm.py
    ${CMAKE_CURRENT_SOURCE_DIR}/im2col.py
    ${CMAKE_CURRENT_SOURCE_DIR}/pool_2d.py
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax.py
    ${CMAKE_CURRENT_SOURCE_DIR}/unary_ops.py
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2p.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2_f32.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2p_f32.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_aie2.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_aie2p.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/relu_aie2.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/relu_aie2p.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/relu_aie2_f32.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/relu_aie2p_f32.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/spec_utils.py
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
        lib/kernels/triton_kernels
    )
