# NOTE: the benchmarks/ subdirectory is added from ../CMakeLists.txt so that the
# matmul benchmark can be built without the HSA backend (e.g. for CPU-only or
# HIP-only configurations). Do not add it here.

#
# test-mul-mat-hsa

set(TEST_TARGET test-mul-mat-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-batch-dispatch-hsa

set(TEST_TARGET test-batch-dispatch-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-vector-hsa

set(TEST_TARGET test-vector-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-event-synchronize-hsa

set(TEST_TARGET test-event-synchronize-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-event-wait-hsa

set(TEST_TARGET test-event-wait-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-backend-ops-mnist

set(TEST_TARGET test-backend-ops-mnist)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-gelu-hsa

set(TEST_TARGET test-gelu-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-broadcast-hsa
#
set(TEST_TARGET test-broadcast-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-norm-hsa
#
set(TEST_TARGET test-norm-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-diag-mask-inf-hsa
#
set(TEST_TARGET test-diag-mask-inf-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
# See the SOFT_MAX block below: DIAG_MASK_INF is reported unsupported by default, so this
# standalone test opts back in rather than skipping every case.
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT
             "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw" "GGML_HSA_ENABLE_FAULTING_OPS=1")

#
# test-unary-hsa
#
set(TEST_TARGET test-unary-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-argmax-hsa
#
set(TEST_TARGET test-argmax-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-get-rows-hsa
#
set(TEST_TARGET test-get-rows-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-convert-hsa
#
set(TEST_TARGET test-convert-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-convert-pad-hsa
#
set(TEST_TARGET test-convert-pad-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# test-depad-hsa
#
set(TEST_TARGET test-depad-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")
#
# test-softmax-hsa
#
set(TEST_TARGET test-softmax-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
# SOFT_MAX is reported unsupported by default (it faults the AIE queue when run back to back
# inside a full attention graph). This test dispatches it alone, which does not, so opt back in --
# otherwise every case would skip and the suite would pass without testing the kernel.
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT
             "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw" "GGML_HSA_ENABLE_FAULTING_OPS=1")
#
# test-preprocess-hsa
#
set(TEST_TARGET test-preprocess-hsa)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-hsa)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

#
# Kernel-inlining variants
#
# GGML_HSA_KERNEL_INLINE=1 hands each kernel to aiecc as textual LLVM IR and IR-links it into the
# core module, instead of linking a .o with ld.lld. That changes code generation -- notably it
# moves the kernel's frame onto the core's stack -- so a kernel that builds one way can fail to
# build the other. Nothing else in the build or in CI sets the variable, so without these entries
# the flag ships untested: it was added broken for every binary element-wise kernel and no test
# went red.
#
# These reuse the existing executables; only the environment differs. One entry per IRON kernel
# module that opts in to inlining.
#
# GGML_HSA_KERNEL_CACHE_DIR is not optional here. The kernel cache is keyed by op, shapes and
# dtypes -- it does not record whether a kernel was inlined -- so the two modes alias. Sharing the
# default cache would let whichever variant ran first serve its artifacts to the other, and the
# inline variants would silently test nothing. A separate directory also keeps these independent
# of test execution order, rather than having them clear a cache the other tests are using.
set(GGML_HSA_INLINE_TESTS
    test-argmax-hsa     # argmax.py
    test-broadcast-hsa  # binary_ops.py, broadcast and row paths
    test-depad-hsa      # depad.py
    test-gelu-hsa       # unary_ops.py, the largest unary frame
    test-mul-mat-hsa    # gemm.py
    test-unary-hsa      # unary_ops.py
    test-vector-hsa     # binary_ops.py, element-wise path
)

foreach(TEST_TARGET IN LISTS GGML_HSA_INLINE_TESTS)
    add_test(NAME ${TEST_TARGET}-inline COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET}-inline PROPERTY ENVIRONMENT
                 "LLVM_PROFILE_FILE=${TEST_TARGET}-inline.profraw"
                 "GGML_HSA_KERNEL_INLINE=1"
                 "GGML_HSA_KERNEL_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/inline-kernel-cache")
endforeach()
