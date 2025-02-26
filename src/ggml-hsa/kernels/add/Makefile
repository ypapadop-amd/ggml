##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

targetname = add
devicename = npu
col = 0

all: build/${targetname}.pdi

build/${targetname}.mlir: ${targetname}.py
	mkdir -p ${@D}
	python3 $< ${devicename} ${col} > $@

build/${targetname}.xclbin: build/${targetname}.mlir
	cd ${@D} && \
	aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=${targetname}_insts.txt ${<F}

build/${targetname}.pdi: build/${targetname}.xclbin
	cd ${@D} && \
	xclbinutil --dump-section AIE_PARTITION:JSON:aie_partition.json --force --input ${<F} && \
	PDI_FILENAME=$$(jq .aie_partition.PDIs.[0].file_name aie_partition.json) && \
	bash -c "cp $$PDI_FILENAME ${@F}"

convert: build/aie_partition.json
	cd build && jq .aie_partition.PDIs.[0].file_name aie_partition.json

clean:
	rm -rf build *.pdi
