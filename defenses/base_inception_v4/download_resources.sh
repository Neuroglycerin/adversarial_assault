#!/bin/bash
#
# Script to download checkpoints and other resources for the model.
#

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

if [ ! -e "inception_v4.ckpt" ]; then
    # Download inception v4 checkpoint
    wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
    tar -xvzf inception_v4_2016_09_09.tar.gz
    rm inception_v4_2016_09_09.tar.gz
fi

