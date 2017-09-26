#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Download dataset.
mkdir -p dataset/images
python dataset/download_images.py \
  --input_file=dataset/dev_dataset.csv \
  --output_dir=dataset/images/
