#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

CONFIG="configs/setfit.yaml"

echo "Running SetFit baseline..."
python -m src.setfit_baseline --config $CONFIG --fold 0

echo ""
echo "SetFit baseline complete!"
