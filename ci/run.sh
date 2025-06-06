#/bin/bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # With SYCL support
# GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -v $OUT/*.log
rm -v $OUT/*.exit
rm -v $OUT/*.md

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

CMAKE_EXTRA=""
CTEST_EXTRA=""

if [ ! -z ${GG_BUILD_CUDA} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_CUDA=ON"
fi

if [ ! -z ${GG_BUILD_METAL} ]; then
    # TODO: this should use -DGGML_METAL_SHADER_DEBUG=ON instead, but currently it fails because
    #       the binaries cannot locate default.metallib eventhough it is in bin/. cannot figure out
    #       why this is happening, so temporary workaround is to use -DGGML_METAL_EMBED_LIBRARY=ON
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON"
fi

if [ ! -z ${GG_BUILD_SYCL} ]; then
    if [ -z ${ONEAPI_ROOT} ]; then
        echo "Not detected ONEAPI_ROOT, please install oneAPI base toolkit and enable it by:"
        echo "source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    export ZES_ENABLE_SYSMAN=1
    # No plan to implement backward pass for now / disable test-opt
    CTEST_EXTRA="-E test-opt"
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_SYCL=1 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON"
fi

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -N $url

    cd $cwd
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

function gg_run {
    ci=$1

    set -o pipefail
    set -x

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +x
    set +o pipefail

    gg_sum_$ci

    ret=$((ret | cur))
}

## ci

# ctest_debug

function gg_run_ctest_debug {
    cd ${SRC}

    rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} ..     ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                                              ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ ! -z ${GG_BUILD_METAL} ]; then
        export GGML_METAL_PATH_RESOURCES="$(pwd)/bin"
    fi

    (time ctest ${CTEST_EXTRA} --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

# ctest_release

function gg_run_ctest_release {
    cd ${SRC}

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} ..   ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                                              ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ ! -z ${GG_BUILD_METAL} ]; then
        export GGML_METAL_PATH_RESOURCES="$(pwd)/bin"
    fi

    if [ -z $GG_BUILD_LOW_PERF ]; then
        (time ctest ${CTEST_EXTRA} --output-on-failure ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest ${CTEST_EXTRA} --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    fi

    set +e
}

function gg_sum_ctest_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# gpt_2

function gg_run_gpt_2 {
    cd ${SRC}

    gg_wget models-mnt/gpt-2 https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-117M.bin

    cd build-ci-release

    set -e

    model="../models-mnt/gpt-2/ggml-model-gpt-2-117M.bin"
    prompts="../examples/prompts/gpt-2.txt"

    (time ./bin/gpt-2-backend --model ${model} -s 1234 -n 64 -tt ${prompts}                       ) 2>&1 | tee -a $OUT/${ci}-tg.log
    (time ./bin/gpt-2-backend --model ${model} -s 1234 -n 64 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg.log
    (time ./bin/gpt-2-sched   --model ${model} -s 1234 -n 64 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg.log

    (time ./bin/gpt-2-batched --model ${model} -s 1234 -n 64 -np 8 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg.log

    set +e
}

function gg_sum_gpt_2 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs short GPT-2 text generation\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-tg.log)"
    gg_printf '```\n'
}

# TODO: update
## mnist
#
#function gg_run_mnist {
#    cd ${SRC}
#
#    cd build-ci-release
#
#    set -e
#
#    mkdir -p models/mnist
#    python3 ../examples/mnist/convert-h5-to-ggml.py ../examples/mnist/models/mnist/mnist_model.state_dict
#
#    model_f32="./models/mnist/ggml-model-f32.bin"
#    samples="../examples/mnist/models/mnist/t10k-images.idx3-ubyte"
#
#    # first command runs and exports "mnist.ggml", the second command runs the exported model
#
#    (time ./bin/mnist     ${model_f32} ${samples} ) 2>&1 | tee -a $OUT/${ci}-mnist.log
#    (time ./bin/mnist-cpu ./mnist.ggml ${samples} ) 2>&1 | tee -a $OUT/${ci}-mnist.log
#
#    set +e
#}
#
#function gg_sum_mnist {
#    gg_printf '### %s\n\n' "${ci}"
#
#    gg_printf 'MNIST\n'
#    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
#    gg_printf '```\n'
#    gg_printf '%s\n' "$(cat $OUT/${ci}-mnist.log)"
#    gg_printf '```\n'
#}

# sam

function gg_run_sam {
    cd ${SRC}

    gg_wget models-mnt/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    gg_wget models-mnt/sam/ https://raw.githubusercontent.com/YavorGIvanov/sam.cpp/ceafb7467bff7ec98e0c4f952e58a9eb8fd0238b/img.jpg

    cd build-ci-release

    set -e

    path_models="../models-mnt/sam/"
    model_f16="${path_models}/ggml-model-f16.bin"
    img_0="${path_models}/img.jpg"

    python3 ../examples/sam/convert-pth-to-ggml.py ${path_models}/sam_vit_b_01ec64.pth ${path_models}/ 1

    # Test default parameters
    (time ./bin/sam -m ${model_f16} -i ${img_0} ) 2>&1 | tee -a $OUT/${ci}-main.log
    grep -q "point prompt" $OUT/${ci}-main.log
    grep -q "bbox (371, 436), (144, 168)" $OUT/${ci}-main.log

    # Test box prompt and single mask output
    (time ./bin/sam -m ${model_f16} -i ${img_0} -b 368,144,441,173 -sm) 2>&1 | tee -a $OUT/${ci}-main.log
    grep -q "box prompt" $OUT/${ci}-main.log
    grep -q "bbox (370, 439), (144, 169)" $OUT/${ci}-main.log

    set +e
}

function gg_sum_sam {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Run SAM\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-main.log)"
    gg_printf '```\n'
}

# yolo

function gg_run_yolo {
    cd ${SRC}

    gg_wget models-mnt/yolo/ https://huggingface.co/ggml-org/models/resolve/main/yolo/yolov3-tiny.weights
    gg_wget models-mnt/yolo/ https://huggingface.co/ggml-org/models/resolve/main/yolo/dog.jpg

    cd build-ci-release
    cp -r ../examples/yolo/data .

    set -e

    path_models="../models-mnt/yolo/"

    python3 ../examples/yolo/convert-yolov3-tiny.py ${path_models}/yolov3-tiny.weights

    (time ./bin/yolov3-tiny -m yolov3-tiny.gguf -i ${path_models}/dog.jpg ) 2>&1 | tee -a $OUT/${ci}-main.log

    grep -qE "dog: (55|56|57|58|59)%" $OUT/${ci}-main.log
    grep -qE "car: (50|51|52|53|54)%" $OUT/${ci}-main.log
    grep -qE "truck: (54|55|56|57|58)%" $OUT/${ci}-main.log
    grep -qE "bicycle: (57|58|59|60|61)%" $OUT/${ci}-main.log

    set +e
}

function gg_sum_yolo {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Run YOLO\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-main.log)"
    gg_printf '```\n'
}

## main

if [ -z ${GG_BUILD_LOW_PERF} ]; then
    # Create symlink: ./ggml/models-mnt -> $MNT/models/models-mnt
    rm -rf ${SRC}/models-mnt
    mnt_models=${MNT}/models
    mkdir -p ${mnt_models}
    ln -sfn ${mnt_models} ${SRC}/models-mnt

    # Create a fresh python3 venv and enter it
    if ! python3 -m venv "$MNT/venv"; then
        echo "Error: Failed to create Python virtual environment at $MNT/venv."
        exit 1
    fi
    source "$MNT/venv/bin/activate"

    pip install -r ${SRC}/requirements.txt --disable-pip-version-check
fi


ret=0
if [ -z ${GG_BUILD_SYCL}]; then
    test $ret -eq 0 && gg_run ctest_debug
fi
test $ret -eq 0 && gg_run ctest_release

if [ ! -z ${GG_BUILD_METAL} ]; then
    export GGML_METAL_PATH_RESOURCES="${SRC}/build-ci-release/bin"
fi

if [ -z ${GG_BUILD_NO_DOWNLOAD} ]; then
    test $ret -eq 0 && gg_run gpt_2
    #test $ret -eq 0 && gg_run mnist
    test $ret -eq 0 && gg_run sam
    test $ret -eq 0 && gg_run yolo
fi

if [ -z $GG_BUILD_LOW_PERF ]; then
    if [ -z ${GG_BUILD_VRAM_GB} ] || [ ${GG_BUILD_VRAM_GB} -ge 16 ]; then
        # run tests that require GPU with at least 16GB of VRAM
        date
    fi
fi

exit $ret
