#!/bin/bash
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

SCRIPT_DIR_NAME=$(dirname -- "${BASH_SOURCE[0]}")
VENV_NAME=.venv

# Parse backend argument (comma-separated list, default: iron)
BACKENDS="${1:-iron}"

# Validate and install each backend
IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"
for backend in "${BACKEND_LIST[@]}"; do
    # Trim whitespace
    backend=$(echo "$backend" | xargs)

    REQUIREMENTS_FILE="${SCRIPT_DIR_NAME}/requirements-${backend}.txt"

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "Error: Unknown backend '$backend'"
        echo "Available backends: iron, triton"
        echo "Usage: source $0 [backend1,backend2,...]"
        echo "  Examples:"
        echo "    source $0              # Default: IRON only"
        echo "    source $0 iron         # Explicit IRON"
        echo "    source $0 triton       # Triton (includes IRON)"
        echo "    source $0 iron,triton  # Both (redundant but supported)"
        return 1 2>/dev/null || exit 1
    fi
done

# Create and activate virtual environment
if ! python3 -m venv ${VENV_NAME} || ! source ${VENV_NAME}/bin/activate; then
    echo "Error: could not create or activate virtual environment '${VENV_NAME}'"
    return 1 2>/dev/null || exit 1
fi

# Upgrade pip
if ! python3 -m pip install --upgrade pip; then
    echo "Error: could not upgrade pip"
    return 1 2>/dev/null || exit 1
fi

# If both iron and triton are selected, only install triton requirements
INSTALL_LIST=("${BACKEND_LIST[@]}")
if [[ " ${BACKEND_LIST[*]} " == *" iron "* ]] && [[ " ${BACKEND_LIST[*]} " == *" triton "* ]]; then
    INSTALL_LIST=("triton")
fi

# Install requirements for each backend
for backend in "${INSTALL_LIST[@]}"; do
    backend=$(echo "$backend" | xargs)
    REQUIREMENTS_FILE="${SCRIPT_DIR_NAME}/requirements-${backend}.txt"
    echo "Installing ${backend} backend dependencies from ${REQUIREMENTS_FILE}..."
    if ! python3 -m pip install -r ${REQUIREMENTS_FILE}; then
        echo "Error: could not install ${backend} backend dependencies from ${REQUIREMENTS_FILE}"
        return 1 2>/dev/null || exit 1
    fi
done

# mlir-air pins an mlir-aie older than the hsaco packer that every kernel is packed with, so
# requirements-triton.txt cannot ask for a newer one. Install the build IRON pins over it.
if [[ " ${INSTALL_LIST[*]} " == *" triton "* ]]; then
    echo "Installing the mlir-aie hsaco packer over the Triton dependencies..."
    if ! python3 -m pip install --no-deps \
        --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
        mlir_aie_no_rtti==1.4.5.dev48+gcb6f2d1; then
        echo "Error: could not install the mlir-aie hsaco packer"
        return 1 2>/dev/null || exit 1
    fi
fi

echo "Environment setup complete. Installed backends: ${INSTALL_LIST[*]}"
