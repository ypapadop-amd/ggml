#!/bin/bash
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

SCRIPT_DIR_NAME=$(dirname -- "${BASH_SOURCE[0]}")
VENV_NAME=ggml_hsa_env

python3 -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ${SCRIPT_DIR_NAME}/requirements.txt
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r ${SCRIPT_DIR_NAME}/requirements_extras.txt
