#!/bin/bash
#
# Script to download checkpoints and other resources for the model.
#

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

if [ ! -e "resnet_v1_50.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
    tar -xvzf resnet_v1_50_2016_08_28.tar.gz
    rm resnet_v1_50_2016_08_28.tar.gz
fi

