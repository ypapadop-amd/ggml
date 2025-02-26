##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

kernel = add
dev = npu
dtype = int32_t
size = 256

all: build/${kernel}.pdi

build/${kernel}.mlir: ${kernel}.py
	mkdir -p ${@D}
	python3 $< --dev ${dev} --dtype ${dtype} --size ${size} > $@

build/${kernel}.xclbin: build/${kernel}.mlir
	cd ${@D} && \
	aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=${kernel}_insts.txt ${<F}

build/${kernel}.pdi: build/${kernel}.xclbin
	cd ${@D} && \
	xclbinutil --dump-section AIE_PARTITION:JSON:aie_partition.json --force --input ${<F} && \
	PDI_FILENAME=$$(jq .aie_partition.PDIs.[0].file_name aie_partition.json) && \
	bash -c "cp $$PDI_FILENAME ${@F}"

clean:
	rm -rf build *.pdi
