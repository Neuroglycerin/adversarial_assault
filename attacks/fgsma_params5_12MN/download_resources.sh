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
MODELS_OUT_DIR="${SCRIPT_DIR}/models"

mkdir -p "${MODELS_OUT_DIR}"

cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_100_224 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_100_192 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_100_160 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_100_128 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_075_224 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_075_192 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_075_160 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_075_128 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_050_224 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_050_192 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_050_160 "${MODELS_OUT_DIR}"
cp -r "${MODELS_SOURCE_DIR}"/mobilenet_v1_050_128 "${MODELS_OUT_DIR}"

