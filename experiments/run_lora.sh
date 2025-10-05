#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

CONFIG="configs/lora_modernbert.yaml"

echo "Training 5 folds with LoRA + ASL..."

for fold in {0..4}; do
    echo ""
    echo "========================================="
    echo "Training Fold $fold"
    echo "========================================="
    python -m src.train --config $CONFIG --fold $fold
done

echo ""
echo "Training complete!"
