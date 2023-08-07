#!/bin/sh


echo "Downloading pre-trained pipeline components..."
gdown https://drive.google.com/uc?id=1BZK1yz925oXyNiIyi4UZBBTs6PQKe23W

echo "Unpacking downloaded data..."
unzip -q trained.zip

echo "Finished!"
