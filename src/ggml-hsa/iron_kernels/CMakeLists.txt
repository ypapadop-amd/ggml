#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

include(ggml_hsa_utils)

if (NOT EXISTS $ENV{MLIR_AIE_INSTALL_DIR})
    message(FATAL_ERROR "MLIR_AIE_INSTALL_DIR is not set")
else()
    set(MLIR_AIE_DIR $ENV{MLIR_AIE_INSTALL_DIR})
endif()
list(APPEND CMAKE_MODULE_PATH ${MLIR_AIE_DIR}/usr/lib/cmake/aie)

find_package(Peano REQUIRED)
find_package(AIE REQUIRED)

set(AIE_AIECC "${AIE_TOOLS_BINARY_DIR}/aiecc.py")

# Compile an AIE kernel using IRON.
#
# This function generates PDI and insts files that are set as dependencies to `TARGET`.
#
# Arguments:
#     TARGET_NAME (string): target
#     SOURCE (string): kernel source file
#     DEVICE (string): target device
#     KERNEL_ARGS (string): kernel compilation flags
#     PEANO_SOURCE (string): Peano source file
#     PEANO_OUTPUT_FILE (string): Peano output file name
#     PEANO_COMPILE_ARGS (string): Peano compilation flags
#     EXPORTED_NAME (string): kernel exported name
#
function(add_aie_iron_kernel TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 0 arg
        ""
        "KERNEL_ARGS;SOURCE;DEVICE;PEANO_SOURCE;PEANO_OUTPUT_FILE;EXPORTED_NAME"
        "PEANO_COMPILE_ARGS"
        )

    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../kernels/${arg_DEVICE}")
    set(MLIR_OUTPUT "${OUTPUT_DIR}/${arg_EXPORTED_NAME}.mlir")
    set(INSTS_OUTPUT "${OUTPUT_DIR}/${arg_EXPORTED_NAME}_insts.bin")
    set(PDI_OUTPUT "${OUTPUT_DIR}/${arg_EXPORTED_NAME}.pdi")
    string(REPLACE " " ";" KERNEL_ARGS ${arg_KERNEL_ARGS})

    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    # compile to MLIR
    add_custom_command(
        OUTPUT ${MLIR_OUTPUT}
        COMMAND ${Python_EXECUTABLE} ${arg_SOURCE} "--dev=${arg_DEVICE}" ${KERNEL_ARGS} > ${MLIR_OUTPUT}
        DEPENDS ${arg_SOURCE}
        BYPRODUCTS ${OUTPUT_DIR}/${arg_KERNEL_NAME}.mlir.prj
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMENT "Generating AIE MLIR file for ${arg_KERNEL_NAME}"
        VERBATIM
        )

    set(AIECC_DEPS ${MLIR_OUTPUT})
    if (DEFINED arg_PEANO_SOURCE)
        # compile single-core solution
        set(PEANO_OUTPUT "${OUTPUT_DIR}/${arg_PEANO_OUTPUT_FILE}")

        set(PEANO_WARNING_FLAGS -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body)
        set(PEANO_FLAGS -O2 -std=c++20 --target=aie2-none-unknown-elf ${PEANO_WARNING_FLAGS} -DNDEBUG)

        set(PEANO_COMPILE_DEFS "")
        foreach(flag ${arg_PEANO_COMPILE_ARGS})
            list(APPEND PEANO_COMPILE_DEFS "-D${flag}")
        endforeach()

        add_custom_command(
            OUTPUT ${PEANO_OUTPUT}
            COMMAND ${Peano_CXX} ${PEANO_FLAGS} ${PEANO_COMPILE_DEFS} -I${AIE_INCLUDE_DIRS}
                    -c ${arg_PEANO_SOURCE} -o ${PEANO_OUTPUT}
            DEPENDS ${arg_PEANO_SOURCE}
            WORKING_DIRECTORY ${OUTPUT_DIR}
            COMMENT "Compiling ${arg_PEANO_SOURCE} for ${arg_KERNEL_NAME}"
            VERBATIM
            )

        list(APPEND AIECC_DEPS ${PEANO_OUTPUT})
    endif()

    # compile full solution
    set(AIECC_FLAGS
        --alloc-scheme=basic-sequential --aie-generate-pdi --aie-generate-npu-insts
        --no-compile-host --no-xchesscc --no-xbridge --peano=${Peano_ROOT_DIR})
    add_custom_command(
        OUTPUT ${PDI_OUTPUT} ${INSTS_OUTPUT}
        COMMAND ${Python_EXECUTABLE} ${AIE_AIECC} ${AIECC_FLAGS}
                --pdi-name=${PDI_OUTPUT} --npu-insts-name=${INSTS_OUTPUT} ${MLIR_OUTPUT}
        DEPENDS ${AIECC_DEPS}
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMENT "Generating PDI for ${arg_EXPORTED_NAME}"
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
    add_subdirectory(compiler)

    # copy kernel sources and helper function to build and install directories

    set(IRON_KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/add.py
        ${CMAKE_CURRENT_SOURCE_DIR}/mm.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mul_mat.py
        ${CMAKE_CURRENT_SOURCE_DIR}/zero.cc
        )

    ggml_hsa_copy_files(ggml_hsa_copy_iron_files
        FILES
            ${IRON_KERNEL_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}
        )

    add_dependencies(ggml-hsa ggml_hsa_copy_iron_files)

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
    SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/add.py
    DEVICE
        aie2
    KERNEL_ARGS
        "--tensors (256)/i32 (256)/i32 (256)/i32"
    EXPORTED_NAME
        "add-256i32-256i32-256i32"
    )

add_aie_iron_kernel(ggml-hsa
    SOURCE
       ${CMAKE_CURRENT_SOURCE_DIR}/add.py
    DEVICE
        aie2
    KERNEL_ARGS
        "--tensors (256)/f32 (256)/f32 (256)/f32"
    EXPORTED_NAME
        "add-256f32-256f32-256f32"
    )

add_aie_iron_kernel(ggml-hsa
    SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/add.py
    DEVICE
        aie2
    KERNEL_ARGS
        "--tensors (768)/f32 (768)/f32 (768)/f32"
    EXPORTED_NAME
        "add-768f32-768f32-768f32"
    )

add_aie_iron_kernel(ggml-hsa
    SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/mul_mat.py
    DEVICE
        aie2
    KERNEL_ARGS
        "-M 32 -K 32 -N 32 -m 8 -k 8 -n 8 --dtype_in i16 --dtype_out i16 --n-aie-cols 4 --b-col-maj 0"
    PEANO_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/mm.cc
    PEANO_OUTPUT_FILE
        "mm_8x8x8.o"
    PEANO_COMPILE_ARGS
        i16_i16_ONLY DIM_M=8 DIM_K=8 DIM_N=8
    EXPORTED_NAME
        "mul_mat-32x32i16-32x32i16-32x32i16"
    )
