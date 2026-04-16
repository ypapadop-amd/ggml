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
python3 -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements for each backend
for backend in "${BACKEND_LIST[@]}"; do
    backend=$(echo "$backend" | xargs)
    REQUIREMENTS_FILE="${SCRIPT_DIR_NAME}/requirements-${backend}.txt"
    echo "Installing ${backend} backend dependencies from ${REQUIREMENTS_FILE}..."
    python3 -m pip install -r ${REQUIREMENTS_FILE}
done

echo "Environment setup complete. Installed backends: ${BACKENDS}"
