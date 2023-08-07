#!/bin/sh


echo "Downloading image bounding box data from Lindh et al. (2020)..."
gdown https://drive.google.com/uc?id=1jwgUBnKCCX2ez_n4CREE-5Qfw3PIS-7m

echo "Unpacking downloaded data..."
unzip -q additional_files.zip

echo "Additional setup complete!"
