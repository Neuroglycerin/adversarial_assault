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

cp -r "${CUSTOM_MODELS_SOURCE_DIR}"/inception_v3_5aux_299 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_100_224 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/inception_v1 "${MODELS_OUT_DIR}"
cp -r "${CUSTOM_MODELS_SOURCE_DIR}"/xception_multiscale2_flab_255 "${MODELS_OUT_DIR}"
cp -r "${CUSTOM_MODELS_SOURCE_DIR}"/mobilenet_v1_lahrelu_200_flab_299 "${MODELS_OUT_DIR}"
