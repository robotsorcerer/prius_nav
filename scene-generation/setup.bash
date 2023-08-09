#!/bin/bash

mkdir -p models
cd models
echo "Downloading SAM model..."
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
