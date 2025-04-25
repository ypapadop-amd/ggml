list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../cmake)

find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(Peano REQUIRED)
find_package(MLIR-AIE REQUIRED)

# Copies files to build directory
#
# Arguments:
#     TARGET_NAME (string): target
#     DESTINATION (string): destination directory
#     FILES (string): files to copy
#
function(copy_files_to_build TARGET_NAME)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs FILES)

    cmake_parse_arguments(PARSE_ARGV 0 arg
        "" "${oneValueArgs}" "${multiValueArgs}")

    foreach(FILE IN LISTS arg_FILES)
        get_filename_component(FILE_NAME "${FILE}" NAME)
        set(DESTINATION_FILE "${arg_DESTINATION}/${FILE_NAME}")

        add_custom_command(
            OUTPUT ${DESTINATION_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILE} ${DESTINATION_FILE}
            DEPENDS ${FILE}
            COMMENT "Copying ${FILE} to build directory ${arg_DESTINATION}"
            )

        list(APPEND COPIED_FILES "${DESTINATION_FILE}")
    endforeach()

    add_custom_target(copy-files ALL DEPENDS ${COPIED_FILES})
    add_dependencies(${TARGET_NAME} copy-files)
endfunction()


# Compile an AIE kernel using IRON.
#
# This function generates PDI and insts files that are set as dependencies to `TARGET`.
#
# Arguments:
#     TARGET_NAME (string): target
#     KERNEL_NAME (string): kernel name
#     SOURCE (string): kernel source file
#     DEVICE (string): target device
#     KERNEL_ARGS (string): kernel compilation flags
#     PEANO_SOURCE (string): Peano source file
#     PEANO_OUTPUT_FILE (string): Peano output file name
#     PEANO_COMPILE_ARGS (string): Peano compilation flags
#
function(add_aie_iron_kernel TARGET_NAME)
    set(oneValueArgs KERNEL_NAME SOURCE DEVICE PEANO_SOURCE PEANO_OUTPUT_FILE)
    set(multiValueArgs KERNEL_ARGS PEANO_COMPILE_ARGS)

    cmake_parse_arguments(PARSE_ARGV 0 arg
        "" "${oneValueArgs}" "${multiValueArgs}")

    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${arg_DEVICE}")
    set(MLIR_OUTPUT "${OUTPUT_DIR}/${arg_KERNEL_NAME}.mlir")
    set(INSTS_OUTPUT "${OUTPUT_DIR}/${arg_KERNEL_NAME}_insts.bin")
    set(PDI_OUTPUT "${OUTPUT_DIR}/${arg_KERNEL_NAME}.pdi")

    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    add_custom_command(
        OUTPUT ${MLIR_OUTPUT}
        COMMAND ${Python_EXECUTABLE} ${arg_SOURCE} ${arg_KERNEL_ARGS} > ${MLIR_OUTPUT}
        DEPENDS ${arg_SOURCE}
        BYPRODUCTS ${OUTPUT_DIR}/${arg_KERNEL_NAME}.mlir.prj
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMENT "Generating AIE MLIR file for ${arg_KERNEL_NAME}"
        VERBATIM
        )

    set(MLIR_AIE_DEPS ${MLIR_OUTPUT})
    if (DEFINED arg_PEANO_SOURCE)
        set(PEANO_OUTPUT "${OUTPUT_DIR}/${arg_PEANO_OUTPUT_FILE}")

        set(PEANO_WARNING_FLAGS -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body)
        set(PEANO_FLAGS -O2 -std=c++20 --target=aie2-none-unknown-elf ${PEANO_WARNING_FLAGS} -DNDEBUG -DBIT_WIDTH=8)

        set(PEANO_COMPILE_DEFS "")
        foreach(flag ${arg_PEANO_COMPILE_ARGS})
            list(APPEND PEANO_COMPILE_DEFS "-D${flag}")
        endforeach()

        add_custom_command(
            OUTPUT ${PEANO_OUTPUT}
            COMMAND ${Peano_CXX} ${PEANO_FLAGS} ${PEANO_COMPILE_DEFS} -I ${MLIR_AIE_INCLUDE_DIR}
                    -c ${arg_PEANO_SOURCE} -o ${PEANO_OUTPUT}
            DEPENDS ${arg_PEANO_SOURCE}
            WORKING_DIRECTORY ${OUTPUT_DIR}
            COMMENT "Compiling ${arg_PEANO_SOURCE} for ${arg_KERNEL_NAME}"
            VERBATIM
            )

        list(APPEND MLIR_AIE_DEPS ${PEANO_OUTPUT})
    endif()

    set(MLIR_AIE_FLAGS
        --alloc-scheme=basic-sequential --aie-generate-pdi --aie-generate-npu-insts
        --no-compile-host --no-xchesscc --no-xbridge --peano=${Peano_ROOT_DIR})

    add_custom_command(
        OUTPUT ${PDI_OUTPUT} ${INSTS_OUTPUT}
        COMMAND ${Python_EXECUTABLE} ${MLIR_AIE_COMPILER} ${MLIR_AIE_FLAGS}
                --pdi-name=${PDI_OUTPUT} --npu-insts-name=${INSTS_OUTPUT} ${MLIR_OUTPUT}
        DEPENDS ${MLIR_AIE_DEPS}
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMENT "Generating PDI for ${arg_KERNEL_NAME}"
        VERBATIM
        )

    add_custom_target(ggml_hsa_kernel_${arg_KERNEL_NAME} ALL DEPENDS ${PDI_OUTPUT} ${INSTS_OUTPUT})
    add_dependencies(${TARGET_NAME} ggml_hsa_kernel_${arg_KERNEL_NAME})

    # copy kernel to install directory
    install(
        FILES
            ${PDI_OUTPUT}
            ${INSTS_OUTPUT}
        DESTINATION
            lib/iron-kernels/${arg_DEVICE}
        )
endfunction()

if (GGML_HSA_JIT_COMPILE)
    # copy kernel sources and helper function to build and install directories
    set(IRON_KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/add.py
        ${CMAKE_CURRENT_SOURCE_DIR}/compile.py
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml_hsa_common.py
        ${CMAKE_CURRENT_SOURCE_DIR}/mm.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mul_mat.py
        ${CMAKE_CURRENT_SOURCE_DIR}/zero.cc
        )

    copy_files_to_build(ggml-hsa
        FILES
            ${IRON_KERNEL_FILES}
        DESTINATION
            ${CMAKE_CURRENT_BINARY_DIR}
        )

    install(
        FILES
            ${IRON_KERNEL_FILES}
        DESTINATION
            lib/iron-kernels
        )
endif ()

#
# Pregenerated kernels

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        "add-256i32-256i32-256i32"
    SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/add.py
    DEVICE
        aie2
    KERNEL_ARGS
        --dev aie2 --dtype i32 --dims 256
    )

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        "add-256f32-256f32-256f32"
    SOURCE
       ${CMAKE_CURRENT_SOURCE_DIR}/add.py
    DEVICE
        aie2
    KERNEL_ARGS
        --dev aie2 --dtype f32 --dims 256
    )

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        "add-768f32-768f32-768f32"
    SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/add.py
    DEVICE
        aie2
    KERNEL_ARGS
        --dev aie2 --dtype f32 --dims 768
    )

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        "mul_mat-32x32i16-32x32i16-32x32i16"
    SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/mul_mat.py
    DEVICE
        aie2
    KERNEL_ARGS
        -M 32 -K 32 -N 32 -m 8 -k 8 -n 8 --dtype_in i16 --dtype_out i16 --n-aie-cols 4 --b-col-maj 0
    PEANO_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/mm.cc
    PEANO_OUTPUT_FILE
        "mm_8x8x8.o"
    PEANO_COMPILE_ARGS
        i16_i16_ONLY DIM_M=8 DIM_K=8 DIM_N=8
    )
