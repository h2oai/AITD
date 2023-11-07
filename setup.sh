#!/bin/bash

set -e

# # Create Virtual Environmemt 
eval "$(conda shell.bash hook)"
conda create -n aitd python=3.10 -y
conda activate aitd
pip install -r requirements.txt

# download models
mkdir -p models/toxicity_model && cd models/toxicity_model && curl -L --output toxic_bias-4e693588.ckpt https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_bias-4e693588.ckpt  && cd ../..
mkdir -p models/toxicity_model/transformers && cd models/toxicity_model/transformers && git clone --depth 1 https://huggingface.co/roberta-base && cd ../../..
mkdir -p models/transformers && cd models/transformers && git clone --depth 1 https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion && cd ../..]
