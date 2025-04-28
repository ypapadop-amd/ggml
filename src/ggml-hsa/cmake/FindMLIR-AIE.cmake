#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

find_path(MLIR_AIE_INCLUDE_DIR
    NAMES
        version.h
    PATHS
        $ENV{MLIR_AIE_INSTALL_DIR}/include/aie
    DOC
        "Path to MLIR-AIE headers"
    NO_DEFAULT_PATH
)

find_file(MLIR_AIE_COMPILER
    NAMES
        aiecc.py
    PATHS
        $ENV{MLIR_AIE_INSTALL_DIR}/bin
    DOC
        "Path to MLIR-AIE compiler"
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLIR-AIE
    FOUND_VAR MLIR-AIE_FOUND
    REQUIRED_VARS MLIR_AIE_INCLUDE_DIR MLIR_AIE_COMPILER
)

mark_as_advanced(MLIR_AIE_INCLUDE_DIR MLIR_AIE_COMPILER)
