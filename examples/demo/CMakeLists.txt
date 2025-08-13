# test kernel to get working and built

set(TEST_TARGET hg-tutorial-simple-ctx)
add_executable(${TEST_TARGET} hg-tutorial-simple-ctx)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

# my test kernel for matrix-mul

set(TEST_TARGET my-test-kernel)
add_executable(${TEST_TARGET} my-test-kernel)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

# test kernel for connecting graphs

set(TEST_TARGET connecting-graphs-kernel)
add_executable(${TEST_TARGET} connecting-graphs-kernel)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

# test kernel for moving data

#set(TEST_TARGET my-test-kernel)
#add_executable(${TEST_TARGET} my-test-kernel)
#target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (GGML_HSA)
    add_compile_definitions(GGML_USE_HSA)
endif()

