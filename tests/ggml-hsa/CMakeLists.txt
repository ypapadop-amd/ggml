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
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")
