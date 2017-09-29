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

if hash nvidia-docker 2>/dev/null; then
  echo "Detected GPU is available."
  GPU_FLAG="--gpu"
else
  echo "No GPU detected. Will run on CPU."
  GPU_FLAG="--nogpu"
fi

# Prepare working directory and copy all necessary files.
# In particular copy attacks defenses and dataset, so originals won't
# be overwritten.
WORKING_DIR="$(dirname "${SCRIPT_DIR}")"/100working_"${MAX_EPSILON}"

OUTPUT_DIR=${2:-"$(dirname "${SCRIPT_DIR}")"/100output}
mkdir -p "${OUTPUT_DIR}"

OUTPUT_FILE="${OUTPUT_DIR}/duration_attack.csv"
touch "${OUTPUT_FILE}"

echo "Doing untargeted attacks"
for attack in "${ATTACKS_DIR}"/*; do
    if [ ! -d attack ]; then
        continue
    fi
    echo "Preparing working directory: ${WORKING_DIR}"
    sudo rm -rf "${WORKING_DIR}"
    mkdir -p "${WORKING_DIR}"
    mkdir -p "${WORKING_DIR}/attacks"
    touch "${WORKING_DIR}/attacks/placeholder"
    mkdir -p "${WORKING_DIR}/targeted_attacks"
    touch "${WORKING_DIR}/targeted_attacks/placeholder"
    mkdir -p "${WORKING_DIR}/dataset"
    cp -R "${DATASET_DIR}"/* "${WORKING_DIR}/dataset"
    cp "${DATASET_METADATA_FILE}" "${WORKING_DIR}/dataset.csv"

    cp -R "${ATTACKS_DIR}/${attack}" "${WORKING_DIR}/attacks"

    echo "Running ${attack} on 100 sample images for the first time"
    python "${SCRIPT_DIR}/run_attack_100.py" \
        --attacks_dir="${WORKING_DIR}/attacks" \
        --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
        --dataset_dir="${WORKING_DIR}/dataset" \
        --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
        --dataset_metadata="${WORKING_DIR}/dataset.csv" \
        --output_dir="${WORKING_DIR}/output_dir" \
        --epsilon="${MAX_EPSILON}" \
        ${GPU_FLAG}

    sudo rm -rf "${WORKING_DIR}/intermediate_results"
    sudo rm -rf "${WORKING_DIR}/output"

    echo "Running ${attack} on 100 sample images for the second time"
    python "${SCRIPT_DIR}/run_attack_100.py" \
        --attacks_dir="${WORKING_DIR}/attacks" \
        --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
        --dataset_dir="${WORKING_DIR}/dataset" \
        --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
        --dataset_metadata="${WORKING_DIR}/dataset.csv" \
        --output_dir="${WORKING_DIR}/output_dir" \
        --epsilon="${MAX_EPSILON}" \
        ${GPU_FLAG}

    tail -1 "${WORKING_DIR}/output/duration_attack.csv" >> "${OUTPUT_FILE}"
done

echo "Output is saved in '${OUTPUT_FILE}'"


OUTPUT_FILE="${OUTPUT_DIR}/duration_targeted_attack.csv"
touch "${OUTPUT_FILE}"

echo "Doing targeted attacks"
for attack in ${ATTACKS_DIR}/*; do
    if [ ! -d attack ]; then
        continue
    fi
    echo "Preparing working directory: ${WORKING_DIR}"
    sudo rm -rf "${WORKING_DIR}"
    mkdir -p "${WORKING_DIR}"

    mkdir -p "${WORKING_DIR}/attacks"
    touch "${WORKING_DIR}/attacks/placeholder"
    mkdir -p "${WORKING_DIR}/targeted_attacks"
    touch "${WORKING_DIR}/targeted_attacks/placeholder"
    mkdir -p "${WORKING_DIR}/dataset"
    cp -R "${DATASET_DIR}"/* "${WORKING_DIR}/dataset"
    cp "${DATASET_METADATA_FILE}" "${WORKING_DIR}/dataset.csv"

    cp -R "${TARGETED_ATTACKS_DIR}/${attack}" "${WORKING_DIR}/targeted_attacks"

    echo "Running ${attack} on 100 sample images for the first time"
    python "${SCRIPT_DIR}/run_attack_100.py" \
        --attacks_dir="${WORKING_DIR}/attacks" \
        --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
        --dataset_dir="${WORKING_DIR}/dataset" \
        --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
        --dataset_metadata="${WORKING_DIR}/dataset.csv" \
        --output_dir="${WORKING_DIR}/output_dir" \
        --epsilon="${MAX_EPSILON}" \
        ${GPU_FLAG}

    sudo rm -rf "${WORKING_DIR}/intermediate_results"
    sudo rm -rf "${WORKING_DIR}/output"

    echo "Running ${attack} on 100 sample images for the second time"
    python "${SCRIPT_DIR}/run_attack_100.py" \
        --attacks_dir="${WORKING_DIR}/attacks" \
        --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
        --dataset_dir="${WORKING_DIR}/dataset" \
        --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
        --dataset_metadata="${WORKING_DIR}/dataset.csv" \
        --output_dir="${WORKING_DIR}/output_dir" \
        --epsilon="${MAX_EPSILON}" \
        ${GPU_FLAG}

    tail -1 "${WORKING_DIR}/output/duration_targeted_attack.csv" >> "${OUTPUT_FILE}"
done

echo "Output is saved in '${OUTPUT_FILE}'"
