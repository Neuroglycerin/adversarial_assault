#!/bin/bash

# exit on first error
set -e

MAX_EPSILON="$1"

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ATTACKS_DIR="${SCRIPT_DIR}/attacks"
TARGETED_ATTACKS_DIR="${SCRIPT_DIR}/targeted_attacks"
DATASET_DIR="${SCRIPT_DIR}/dataset/100images"
DATASET_METADATA_FILE="${SCRIPT_DIR}/dataset/dev_dataset_100.csv"

mkdir -p "${ATTACKS_DIR}"
mkdir -p "${TARGETED_ATTACKS_DIR}"
touch "${ATTACKS_DIR}/placeholder"
touch "${TARGETED_ATTACKS_DIR}/placeholder"

# Prepare working directory and copy all necessary files.
# In particular copy attacks defenses and dataset, so originals won't
# be overwritten.
WORKING_DIR=${2:-"$(dirname "${SCRIPT_DIR}")"/100working_"${MAX_EPSILON}"}
mkdir -p "${WORKING_DIR}"

echo "Preparing working directory: ${WORKING_DIR}"
sudo rm -rf "${WORKING_DIR}/attacks"
sudo rm -rf "${WORKING_DIR}/targeted_attacks"
sudo rm -rf "${WORKING_DIR}/intermediate_results/all_adv_examples/"
mkdir -p "${WORKING_DIR}/attacks"
mkdir -p "${WORKING_DIR}/targeted_attacks"
mkdir -p "${WORKING_DIR}/dataset"
mkdir -p "${WORKING_DIR}/intermediate_results"
mkdir -p "${WORKING_DIR}/output_dir"
cp -R "${ATTACKS_DIR}"/* "${WORKING_DIR}/attacks"
cp -R "${TARGETED_ATTACKS_DIR}"/* "${WORKING_DIR}/targeted_attacks"
cp -R "${DATASET_DIR}"/* "${WORKING_DIR}/dataset"
cp "${DATASET_METADATA_FILE}" "${WORKING_DIR}/dataset.csv"

if hash nvidia-docker 2>/dev/null; then
  echo "Detected GPU is available."
  GPU_FLAG="--gpu"
else
  echo "No GPU detected. Will run on CPU."
  GPU_FLAG="--nogpu"
fi

echo "Running attacks on 100 sample images"
python "${SCRIPT_DIR}/run_attack_100.py" \
  --attacks_dir="${WORKING_DIR}/attacks" \
  --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
  --dataset_dir="${WORKING_DIR}/dataset" \
  --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
  --dataset_metadata="${WORKING_DIR}/dataset.csv" \
  --output_dir="${WORKING_DIR}/output_dir" \
  --epsilon="${MAX_EPSILON}" \
  ${GPU_FLAG}

echo "Output is saved in directory '${WORKING_DIR}/output_dir'"
