include(FindPackageHandleStandardArgs)

find_program(Peano_EXECUTABLE
    NAMES
        clang++
    PATHS
        $ENV{PEANO_INSTALL_DIR}/bin
    DOC
        "Path to Peano"
    NO_DEFAULT_PATH
    REQUIRED
)

find_package_handle_standard_args(Peano REQUIRED_VARS Peano_EXECUTABLE)

if (Peano_FOUND)
    mark_as_advanced(Peano_EXECUTABLE)
    set(Peano_ROOT $ENV{PEANO_INSTALL_DIR})
endif()
