#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Creates a target with name TARGET_NAME which copies files to build directory
#
# Arguments:
#     TARGET_NAME (string): target
#     DESTINATION (string): destination directory
#     FILES (string): files to copy
#
function(ggml_hsa_copy_files TARGET_NAME)
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

    add_custom_target(${TARGET_NAME} ALL DEPENDS ${COPIED_FILES})
endfunction()
