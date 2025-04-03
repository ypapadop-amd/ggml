list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../cmake)

find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(Peano REQUIRED)
find_package(MLIR-AIE REQUIRED)

# Compile an AIE kernel using IRON.
#
# This function generates PDI and insts files that are set as dependencies to `TARGET`.
#
# Arguments:
#     TARGET_NAME (string): target
#     SOURCE (string): kernel source file
#     KERNEL_NAME (string): kernel name
#     DEVICE (string): target device
#     KERNEL_SUFFIX (string): kernel name suffix
#     KERNEL_ARGS (string): kernel compilation flags
#     OBJECT_FILE (string): Peano output file
#     COMPILE_ARGS (string): Peano compilation flags
#
function(add_aie_iron_kernel TARGET_NAME)
    set(options)
    set(oneValueArgs SOURCE KERNEL_NAME DEVICE KERNEL_SUFFIX OBJECT_FILE)
    set(multiValueArgs KERNEL_ARGS COMPILE_ARGS)

    cmake_parse_arguments(PARSE_ARGV 0 arg
        "${options}" "${oneValueArgs}" "${multiValueArgs}")

    set(KERNEL_NAME "${arg_KERNEL_NAME}-${arg_DEVICE}-${arg_KERNEL_SUFFIX}")
    set(KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${arg_SOURCE}")
    set(MLIR_OUTPUT "${KERNEL_NAME}.mlir")
    set(XCLBIN_OUTPUT "${KERNEL_NAME}.xclbin")
    set(INSTS_OUTPUT "${KERNEL_NAME}_insts.bin")
    set(PARTITION_OUTPUT "${KERNEL_NAME}_partition.json")
    set(PDI_OUTPUT "${KERNEL_NAME}.pdi")

    add_custom_command(
        OUTPUT ${MLIR_OUTPUT}
        COMMAND ${Python_EXECUTABLE} ${KERNEL_SOURCE} ${arg_KERNEL_ARGS} > ${MLIR_OUTPUT}
        DEPENDS ${KERNEL_SOURCE}
        BYPRODUCTS ${KERNEL_NAME}.mlir.prj
        COMMENT "Generating AIE MLIR file for ${KERNEL_NAME} in ${MLIR_OUTPUT}"
        VERBATIM
    )

    if (DEFINED arg_OBJECT_FILE)
        set(PEANO_WARNING_FLAGS -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body)
        set(PEANO_FLAGS -O2 -std=c++20 --target=aie2-none-unknown-elf ${PEANO_WARNING_FLAGS} -DNDEBUG -DBIT_WIDTH=8)
        set(PEANO_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${arg_SOURCE}")
        string(REPLACE ".py" ".cc" PEANO_SOURCE ${PEANO_SOURCE})
        set(PEANO_OUTPUT "${arg_OBJECT_FILE}")

        set(PEANO_COMPILE_DEFS "")
        foreach(device_flag ${arg_COMPILE_ARGS})
            list(APPEND PEANO_COMPILE_DEFS "-D${device_flag}")
        endforeach()

        add_custom_command(
            OUTPUT ${PEANO_OUTPUT}
            COMMAND ${Peano_EXECUTABLE} ${PEANO_FLAGS} ${PEANO_COMPILE_DEFS} -I ${MLIR_AIE_INCLUDE_DIR}
                    -c ${PEANO_SOURCE} -o ${PEANO_OUTPUT}
            DEPENDS ${PEANO_SOURCE}
            COMMENT "Compiling ${PEANO_SOURCE} for ${KERNEL_NAME} in ${PEANO_OUTPUT}"
            VERBATIM
        )

        add_custom_command(
            OUTPUT ${PDI_OUTPUT} ${INSTS_OUTPUT}
            COMMAND aiecc.py --alloc-scheme=basic-sequential --aie-generate-pdi --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge
                    --peano=${Peano_ROOT}
                    --pdi-name=${PDI_OUTPUT} --npu-insts-name=${INSTS_OUTPUT} ${MLIR_OUTPUT}
            DEPENDS ${PEANO_OUTPUT} ${MLIR_OUTPUT}
            COMMENT "Generating PDI for ${KERNEL_NAME} in ${PDI_OUTPUT}"
            VERBATIM
        )
    else ()
        add_custom_command(
            OUTPUT ${PDI_OUTPUT} ${INSTS_OUTPUT}
            COMMAND aiecc.py --aie-generate-pdi --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge
                    --pdi-name=${PDI_OUTPUT} --npu-insts-name=${INSTS_OUTPUT} ${MLIR_OUTPUT}
            DEPENDS ${MLIR_OUTPUT}
            COMMENT "Generating PDI for ${KERNEL_NAME} in ${PDI_OUTPUT}"
            VERBATIM
        )
    endif()

    add_custom_target(ggml_hsa_kernel_${KERNEL_NAME} ALL DEPENDS ${PDI_OUTPUT} ${INSTS_OUTPUT})
    add_dependencies(${TARGET_NAME} ggml_hsa_kernel_${KERNEL_NAME})
    set_property(TARGET ${TARGET_NAME}
        APPEND
        PROPERTY ADDITIONAL_CLEAN_FILES "*.pdi" "core*.elf"
    )

    # copy kernel to install directory
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/${PDI_OUTPUT}
            ${CMAKE_CURRENT_BINARY_DIR}/${INSTS_OUTPUT}
        DESTINATION lib/kernels/
    )
endfunction()

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        add
    SOURCE
        add.py
    DEVICE
        aie2
    KERNEL_SUFFIX
        "256i32-256i32-256i32"
    KERNEL_ARGS
        --dev aie2 --dtype i32 --dims 256
)

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        add
    SOURCE
        add.py
    DEVICE
        aie2
    KERNEL_SUFFIX
        "256f32-256f32-256f32"
    KERNEL_ARGS
        --dev aie2 --dtype f32 --dims 256
)

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        add
    SOURCE
        add.py
    DEVICE
        aie2
    KERNEL_SUFFIX
        "768f32-768f32-768f32"
    KERNEL_ARGS
        --dev aie2 --dtype f32 --dims 768
)

add_aie_iron_kernel(ggml-hsa
    KERNEL_NAME
        mul_mat
    SOURCE
        mul_mat.py
    DEVICE
        aie2
    KERNEL_SUFFIX
        "32x32i16-32x32i16-32x32i16"
    KERNEL_ARGS
        -M 32 -K 32 -N 32 -m 8 -k 8 -n 8 --dtype_in i16 --dtype_out i16 --n-aie-cols 4 --b-col-maj 0

    OBJECT_FILE
        "mm_8x8x8.o"
    COMPILE_ARGS
        i16_i16_ONLY DIM_M=8 DIM_K=8 DIM_N=8
)
