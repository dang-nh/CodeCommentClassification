#!/bin/bash

echo "Tuning thresholds for all folds..."

for fold in {0..4}; do
    echo ""
    echo "Tuning thresholds for Fold $fold..."
    python -m src.thresholding \
        --preds runs/fold_${fold}/val_preds.npy \
        --labels runs/fold_${fold}/val_labels.npy \
        --out runs/fold_${fold}/thresholds.json \
        --label_names data/processed/labels.json
done

echo ""
echo "Threshold tuning complete!"
