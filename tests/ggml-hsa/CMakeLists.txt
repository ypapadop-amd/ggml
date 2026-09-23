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
