#!/bin/bash

# exit on first error
set -e

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ATTACKS_DIR="${SCRIPT_DIR}/attacks"
TARGETED_ATTACKS_DIR="${SCRIPT_DIR}/targeted_attacks"
DEFENSES_DIR="${SCRIPT_DIR}/defenses"

for d in "${ATTACKS_DIR}/"*/; do
    if [ -e "$d"download_resources.sh ]; then
        bash "$d"download_resources.sh
    fi
done
for d in "${TARGETED_ATTACKS_DIR}/"*/; do
    if [ -e "$d"download_resources.sh ]; then
        bash "$d"download_resources.sh
    fi
done
for d in "${DEFENSES_DIR}/"*/; do
    if [ -e "$d"download_resources.sh ]; then
        bash "$d"download_resources.sh
    fi
done
