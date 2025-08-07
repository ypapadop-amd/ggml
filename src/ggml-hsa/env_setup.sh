#!/bin/bash
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

SCRIPT_DIR_NAME=$(dirname -- "${BASH_SOURCE[0]}")

python3 -m venv ggml_hsa_env
source ggml_hsa_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ${SCRIPT_DIR_NAME}/requirements.txt
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r ${SCRIPT_DIR_NAME}/requirements_extras.txt

MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"

export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${PATH}
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
