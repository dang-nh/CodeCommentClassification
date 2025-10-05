# Reproducing Final Ensemble Results

This document provides the exact commands to reproduce the complete pipeline from scratch.

## Prerequisites

1. **Environment Setup**

```bash
cd /home/team_cv/nhdang/CodeCommentClassification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Data Preparation**

Place your dataset at `data/raw/sentences.csv` with the following columns:
- `id`: Unique sentence identifier
- `class_id`: Group identifier (e.g., file/class name)
- `sentence`: Text to classify
- `lang`: Programming language (JAVA, PY, or PHARO)
- `labels`: Semicolon- or comma-separated label names

## Complete Pipeline (4 Commands)

### Command 1: Create Data Splits

```bash
python -m src.split \
    --input data/raw/sentences.csv \
    --out data/processed/splits.json \
    --test_size 0.2 \
    --folds 5 \
    --seed 42
```

**Output:**
- `data/processed/splits.json` - Contains train/test split and 5-fold CV indices

**Expected:**
- 80% train, 20% test
- 5 folds with no class_id overlap
- Stratified by label distribution

---

### Command 2: Train 5 Folds with LoRA + ASL

```bash
for fold in {0..4}; do
    python -m src.train \
        --config configs/lora_modernbert.yaml \
        --fold $fold
done
```

**Output per fold:**
- `runs/fold_X/best.pt` - Best model checkpoint
- `runs/fold_X/val_preds.npy` - Validation predictions
- `runs/fold_X/val_labels.npy` - Validation ground truth
- `runs/fold_X/train.log` - Training logs
- `runs/fold_X/tensorboard/` - TensorBoard logs

**Expected:**
- Training time: ~30-60 min per fold (depends on dataset size)
- Best validation Micro-F1: >0.80 per fold
- Memory usage: ~18-20GB VRAM

**Monitor training:**
```bash
tensorboard --logdir runs/fold_0/tensorboard
```

---

### Command 3: Tune Per-Label Thresholds

```bash
for fold in {0..4}; do
    python -m src.thresholding \
        --preds runs/fold_${fold}/val_preds.npy \
        --labels runs/fold_${fold}/val_labels.npy \
        --out runs/fold_${fold}/thresholds.json
done
```

**Output per fold:**
- `runs/fold_X/thresholds.json` - Optimal threshold per label

**Expected:**
- Thresholds typically range from 0.3 to 0.7
- Mean threshold around 0.5
- Lower thresholds for rare labels, higher for common labels

---

### Command 4: Run 5-Fold Ensemble on Test Set

```bash
python -m src.inference \
    --config configs/lora_modernbert.yaml \
    --ckpts "runs/fold_*/best.pt" \
    --ensemble mean \
    --out runs/test_preds.csv \
    --test
```

**Output:**
- `runs/test_preds.csv` - Final predictions with probabilities per label
- `runs/test_preds_metrics.json` - Comprehensive evaluation metrics

**Expected metrics:**
```json
{
  "micro_f1": 0.85+,
  "macro_f1": 0.75+,
  "macro_pr_auc": 0.80+,
  "per_label_metrics": {
    "label_name": {
      "precision": 0.XX,
      "recall": 0.XX,
      "f1": 0.XX,
      "pr_auc": 0.XX
    }
  }
}
```

---

## Optional: Generate Visualizations

```bash
python -m src.plotting \
    --preds runs/test_preds.csv \
    --labels data/processed/test_labels.npy \
    --out plots/
```

**Output:**
- `plots/pr_curves_all.png` - PR curves for all labels
- `plots/pr_curve_{label}.png` - Individual PR curve per label
- `plots/label_distribution.png` - Label frequency distribution

---

## Alternative: One-Command Pipeline

Run everything with a single script:

```bash
./run_full_pipeline.sh
```

This executes all 4 commands above sequentially.

---

## Verification Checklist

After running the pipeline, verify:

- [ ] `data/processed/splits.json` exists and contains 5 folds
- [ ] 5 checkpoint files exist: `runs/fold_{0,1,2,3,4}/best.pt`
- [ ] 5 threshold files exist: `runs/fold_{0,1,2,3,4}/thresholds.json`
- [ ] `runs/test_preds.csv` contains predictions for all test samples
- [ ] `runs/test_preds_metrics.json` shows Micro-F1 > 0.80
- [ ] All plots generated in `plots/` directory

---

## Expected Timeline

On a single RTX 3090 / A100 GPU:

| Step | Time | Output |
|------|------|--------|
| 1. Create splits | <1 min | `splits.json` |
| 2. Train 5 folds | 3-5 hours | 5 checkpoints |
| 3. Tune thresholds | <5 min | 5 threshold files |
| 4. Ensemble inference | 5-10 min | Final predictions |
| **Total** | **3-5 hours** | Complete results |

---

## Troubleshooting

### Out of Memory

**Symptom:** CUDA out of memory error during training

**Solution 1:** Reduce batch size
```yaml
# Edit configs/lora_modernbert.yaml
train_params:
  batch_size: 32  # or 24
  grad_accum: 2   # maintain effective batch size
```

**Solution 2:** Reduce sequence length
```yaml
# Edit configs/lora_modernbert.yaml
max_len: 96  # or 64
```

### ModernBERT Not Found

**Symptom:** Model download fails

**Solution:** The code automatically falls back to `microsoft/deberta-v3-base`. To force it:
```yaml
# Edit configs/lora_modernbert.yaml
model_name: "microsoft/deberta-v3-base"
tokenizer_name: "microsoft/deberta-v3-base"
```

### Slow Training

**Symptom:** Training takes >2 hours per fold

**Solution 1:** Ensure bfloat16 is enabled
```yaml
# Edit configs/lora_modernbert.yaml
precision: "bfloat16"  # or "fp16"
```

**Solution 2:** Reduce epochs with early stopping
```yaml
# Edit configs/lora_modernbert.yaml
train_params:
  epochs: 5
logging:
  early_stop: 2
```

### Low Performance

**Symptom:** Micro-F1 < 0.75

**Possible causes:**
1. Insufficient training epochs (increase to 15-20)
2. Learning rate too high/low (try 1e-4 or 3e-4)
3. Class imbalance (tune ASL gamma_neg: 2-6)
4. Data quality issues (check for label noise)

---

## Baseline Comparisons

To compare against baselines:

### SetFit Baseline

```bash
python -m src.setfit_baseline \
    --config configs/setfit.yaml \
    --fold 0
```

**Expected:** Micro-F1 ~0.75-0.80, trains in ~30 min

### TF-IDF Baseline

```bash
python -m src.tfidf_baseline \
    --config configs/tfidf.yaml \
    --fold 0
```

**Expected:** Micro-F1 ~0.65-0.70, trains in ~5 min

---

## Results Summary

After running the complete pipeline, you will have:

1. **Per-label metrics table** showing P/R/F1/PR-AUC for each of ~19 labels
2. **Micro-F1 summary** showing overall performance across all labels
3. **Macro-F1 summary** showing average performance per label
4. **PR curves** visualizing precision-recall tradeoffs
5. **Final predictions CSV** with probabilities for each label

Example output:

```
Results:
  Micro-F1: 0.8542
  Macro-F1: 0.7689
  Macro PR-AUC: 0.8123

Per-label F1 scores:
  label1: 0.89
  label2: 0.82
  label3: 0.76
  ...
```

---

## Next Steps

1. **Analyze results:** Review `runs/test_preds_metrics.json`
2. **Examine errors:** Check predictions where F1 is low
3. **Tune hyperparameters:** Adjust config based on results
4. **Enable chains:** Set `chains.enabled: true` for potential boost
5. **Iterate:** Retrain with improved configuration

---

## Questions?

- Check `README.md` for detailed documentation
- Check `QUICKSTART.md` for quick reference
- Check `PROJECT_SUMMARY.md` for technical details
- Run tests: `python -m tests.test_asl` etc.
