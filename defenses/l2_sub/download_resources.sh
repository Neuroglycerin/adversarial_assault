#!/bin/bash
#
# Script to download checkpoints and other resources for the model.
#

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "${SCRIPT_DIR}/comparison_images" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    mkdir "${SCRIPT_DIR}/comparison_images"
    cp "${SCRIPT_DIR}/../../dataset/images"/*.png "${SCRIPT_DIR}/comparison_images/"
fi
