#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Define the URLs for the checkpoints
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
sam2_hiera_t_url="${BASE_URL}sam2.1_hiera_tiny.pt"
sam2_hiera_s_url="${BASE_URL}sam2.1_hiera_small.pt"
sam2_hiera_b_plus_url="${BASE_URL}sam2.1_hiera_base_plus.pt"
sam2_hiera_l_url="${BASE_URL}sam2.1_hiera_large.pt"


# Download each of the four checkpoints using wget
echo "Downloading sam2_hiera_tiny.pt checkpoint..."
wget $sam2_hiera_t_url -P checkpoints/2.1 || { echo "Failed to download checkpoint from $sam2_hiera_t_url"; exit 1; }

echo "Downloading sam2_hiera_small.pt checkpoint..."
wget $sam2_hiera_s_url -P checkpoints/2.1 || { echo "Failed to download checkpoint from $sam2_hiera_s_url"; exit 1; }

echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
wget $sam2_hiera_b_plus_url -P checkpoints/2.1 || { echo "Failed to download checkpoint from $sam2_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2_hiera_large.pt checkpoint..."
wget $sam2_hiera_l_url -P checkpoints/2.1 || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

echo "All checkpoints are downloaded successfully."


echo "Downloading config files..."

wget https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_t.yaml -P sam2/
wget https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_s.yaml -P sam2/
wget https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_l.yaml -P sam2/
wget https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml -P sam2/

echo "Config files downloaded successfully."
