#!/bin/bash
#
# Script to download checkpoints and other resources for the model.
#

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

if [ -d "slim" ]; then
    rm -rf slim
fi
cp -r ../../tensorflow-slim ./slim


if [ ! -e "inception_v3.ckpt" ]; then
    # Control will enter here if file doesn't exist.
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvzf inception_v3_2016_08_28.tar.gz
    rm inception_v3_2016_08_28.tar.gz
fi


# 70.7
if [ ! -e "mobilenet_v1_1.0_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_224_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_224_2017_06_14.tar.gz
fi

# 69.3
if [ ! -e "mobilenet_v1_1.0_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_192_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_192_2017_06_14.tar.gz
fi

# 67.2
if [ ! -e "mobilenet_v1_1.0_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_160_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_160_2017_06_14.tar.gz
fi

# 64.1
if [ ! -e "mobilenet_v1_1.0_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_128_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_128_2017_06_14.tar.gz
fi


# 68.4
if [ ! -e "mobilenet_v1_0.75_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_224_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_224_2017_06_14.tar.gz
fi

# 67.4
if [ ! -e "mobilenet_v1_0.75_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_192_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_192_2017_06_14.tar.gz
fi

# 65.2
if [ ! -e "mobilenet_v1_0.75_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_160_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_160_2017_06_14.tar.gz
fi

# 61.8
if [ ! -e "mobilenet_v1_0.75_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_128_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_128_2017_06_14.tar.gz
fi


if [ ! -e "mobilenet_v1_0.50_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_224_2017_06_14.tar.gz
    rm mobilenet_v1_0.50_224_2017_06_14.tar.gz
fi

if [ ! -e "mobilenet_v1_0.50_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_192_2017_06_14.tar.gz
    rm mobilenet_v1_0.50_192_2017_06_14.tar.gz
fi

if [ ! -e "mobilenet_v1_0.50_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_160_2017_06_14.tar.gz
    rm mobilenet_v1_0.50_160_2017_06_14.tar.gz
fi

if [ ! -e "mobilenet_v1_0.50_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_128_2017_06_14.tar.gz
    rm mobilenet_v1_0.50_128_2017_06_14.tar.gz
fi

