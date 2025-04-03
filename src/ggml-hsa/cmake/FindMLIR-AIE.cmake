find_path(MLIR_AIE_INCLUDE_DIR
    NAMES
        aie/version.h
    PATHS
        $ENV{MLIR_AIE_INSTALL_DIR}/include
    DOC
        "Path to MLIR-AIE"
    NO_DEFAULT_PATH
    REQUIRED
)

mark_as_advanced(MLIR_AIE_INCLUDE_DIR)
