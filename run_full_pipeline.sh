#!/bin/bash

set -e

echo "========================================="
echo "Code Comment Classification Pipeline"
echo "========================================="
echo ""

echo "Step 1: Creating data splits..."
python -m src.split \
    --input data/raw/sentences.csv \
    --out data/processed/splits.json \
    --test_size 0.2 \
    --folds 5

echo ""
echo "Step 2: Training 5 folds with LoRA + ASL..."
for fold in {0..4}; do
    echo ""
    echo "Training Fold $fold..."
    python -m src.train --config configs/lora_modernbert.yaml --fold $fold
done

echo ""
echo "Step 3: Tuning thresholds for each fold..."
for fold in {0..4}; do
    echo "Tuning thresholds for Fold $fold..."
    python -m src.thresholding \
        --preds runs/fold_${fold}/val_preds.npy \
        --labels runs/fold_${fold}/val_labels.npy \
        --out runs/fold_${fold}/thresholds.json \
        --label_names data/processed/splits.json
done

echo ""
echo "Step 4: Running 5-fold ensemble on test set..."
python -m src.inference \
    --config configs/lora_modernbert.yaml \
    --ckpts "runs/fold_*/best.pt" \
    --ensemble mean \
    --out runs/test_preds.csv \
    --test

echo ""
echo "Step 5: Generating PR curves and plots..."
python -m src.plotting \
    --preds runs/test_preds.csv \
    --labels data/processed/test_labels.npy \
    --out plots/

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Predictions: runs/test_preds.csv"
echo "  - Metrics: runs/test_preds_metrics.json"
echo "  - Plots: plots/"
echo ""
