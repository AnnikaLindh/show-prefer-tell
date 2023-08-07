#!/bin/sh


echo "Creating directories..."
mkdir data/cic
mkdir data/cic/raw
mkdir data/cic/labels
mkdir data/cic/splits
mkdir results
mkdir data_exploration
mkdir data_exploration/hist_region_per_caption
mkdir data/region_selector
mkdir data/region_selector/features_iou
mkdir data/region_order
mkdir data/region_order/sinkhorn
mkdir checkpoints

echo "Downloading Bottom-Up model, and CIC captioning model..."
gdown https://drive.google.com/uc?id=1LZTep2eLp2ZM_q0L0qw_3jt0nvr_uR9J

echo "Unpacking downloaded data..."
unzip -q required.zip

echo "Setup complete!"
