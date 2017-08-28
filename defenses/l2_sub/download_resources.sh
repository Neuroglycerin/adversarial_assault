#!/bin/bash
#
# Script to download checkpoints and other resources for the model.
#

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

if [ ! -e "inception_v3.ckpt" ]; then
    # Control will enter here if file doesn't exist.
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvzf inception_v3_2016_08_28.tar.gz
    rm inception_v3_2016_08_28.tar.gz
fi

if [ ! -d "${SCRIPT_DIR}/comparison_images" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    mkdir "${SCRIPT_DIR}/comparison_images"
    cp "${SCRIPT_DIR}/../../dataset/images"/*.png "${SCRIPT_DIR}/comparison_images/"
fi
