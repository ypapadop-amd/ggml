#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

find_program(Peano_CC
    NAMES
        clang
    PATHS
        $ENV{PEANO_INSTALL_DIR}/bin
    DOC
        "Path to Peano C compiler"
    NO_DEFAULT_PATH
)

find_program(Peano_CXX
    NAMES
        clang++
    PATHS
        $ENV{PEANO_INSTALL_DIR}/bin
    DOC
        "Path to Peano C++ compiler"
    NO_DEFAULT_PATH
)

set(Peano_ROOT_DIR $ENV{PEANO_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Peano
    REQUIRED_VARS
        Peano_ROOT_DIR Peano_CC Peano_CXX
)

mark_as_advanced(Peano_ROOT_DIR Peano_CC Peano_CXX)
