#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Download dataset.
mkdir -p dataset/images
python dataset/download_images.py \
  --input_file=dataset/dev_dataset.csv \
  --output_dir=dataset/images/

# Download dataset.
mkdir -p dataset/images
python dataset/download_images.py \
  --input_file=dataset/dev_dataset_100.csv \
  --output_dir=dataset/100images/
