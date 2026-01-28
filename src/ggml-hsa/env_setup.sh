#!/bin/bash
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

SCRIPT_DIR_NAME=$(dirname -- "${BASH_SOURCE[0]}")
VENV_NAME=.venv

python3 -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ${SCRIPT_DIR_NAME}/requirements.txt
