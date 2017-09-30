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

MODELS_SOURCE_DIR=../../models
CUSTOM_MODELS_SOURCE_DIR=../../custom_models
MODELS_OUT_DIR="${SCRIPT_DIR}/models"

mkdir -p "${MODELS_OUT_DIR}"

cp -r "${MODELS_SOURCE_DIR}"/inception_* "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_* "${MODELS_OUT_DIR}"
cp -r "${CUSTOM_MODELS_SOURCE_DIR}"/ens_adv_inception_resnet_v2_5aux "${MODELS_OUT_DIR}"
