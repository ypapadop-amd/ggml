#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

include(ggml_hsa_utils)

# Compile an AIE kernel using IRON.
#
# This function generates PDI and insts files that are set as dependencies to `TARGET`.
#
# Arguments:
#     TARGET_NAME (string): target
#     KERNEL_NAME (string): kernel name
#     KERNEL_SOURCE (string): kernel source file
#     DEVICE (string): target device
#     INPUT_TENSORS (string): input tensor shapes and datatypes
#     OUTPUT_TENSOR (string): output tensor shape and datatype
#     EXPORTED_NAME (string): kernel exported name
#
function(add_aie_iron_kernel TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 0 arg
        ""
        "KERNEL_NAME;KERNEL_SOURCE;DEVICE;OUTPUT_TENSOR;EXPORTED_NAME"
        "INPUT_TENSORS"
        )

    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../kernels/${arg_DEVICE}")
    set(MLIR_OUTPUT "${OUTPUT_DIR}/${arg_EXPORTED_NAME}.mlir")
    set(INSTS_OUTPUT "${OUTPUT_DIR}/${arg_EXPORTED_NAME}_insts.bin")
    set(PDI_OUTPUT "${OUTPUT_DIR}/${arg_EXPORTED_NAME}.pdi")

    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    add_custom_command(
        OUTPUT ${INSTS_OUTPUT} ${PDI_OUTPUT} ${MLIR_OUTPUT}
        COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/compiler.py"
            "--kernel_name" "${arg_KERNEL_NAME}"
            "--kernel_source" "${arg_KERNEL_SOURCE}"
            "--device" "${arg_DEVICE}"
            "--input_tensors" ${arg_INPUT_TENSORS}
            "--output_tensor" "${arg_OUTPUT_TENSOR}"
            "--exported_name" "${arg_EXPORTED_NAME}"
            "--output_directory" "${OUTPUT_DIR}"
        DEPENDS ${arg_KERNEL_SOURCE}
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMENT "Compiling kernel for ${arg_EXPORTED_NAME}"
        VERBATIM
        )

    add_custom_target(ggml_hsa_kernel_${arg_EXPORTED_NAME} ALL DEPENDS ${PDI_OUTPUT} ${INSTS_OUTPUT})
    add_dependencies(${TARGET_NAME} ggml_hsa_kernel_${arg_EXPORTED_NAME})

    # copy kernel to install directory
    install(
        FILES
            ${PDI_OUTPUT}
            ${INSTS_OUTPUT}
        DESTINATION
            lib/kernels/${arg_DEVICE}
        )
endfunction()

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

#
# Pregenerated kernels

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        ggml_op_add
    KERNEL_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/binary_ops.py
    DEVICE
        aie2
    INPUT_TENSORS
        "(256)/i32" "(256)/i32"
    OUTPUT_TENSOR
        "(256)/i32"
    EXPORTED_NAME
        "add-256i32-256i32-256i32"
    )

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        ggml_op_add
    KERNEL_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/binary_ops.py
    DEVICE
        aie2
    INPUT_TENSORS
        "(256)/f32" "(256)/f32"
    OUTPUT_TENSOR
        "(256)/f32"
    EXPORTED_NAME
        "add-256f32-256f32-256f32"
    )

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        ggml_op_mul_mat
    KERNEL_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/mul_mat.py
    DEVICE
        aie2
    INPUT_TENSORS
        "(32,32)/i16" "(32,32)/i16"
    OUTPUT_TENSOR
        "(32,32)/i16"
    EXPORTED_NAME
        "mul_mat-32x32i16-32x32i16-32x32i16"
    )
