# Quick Start Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU with 24GB VRAM (recommended)
- Dataset: `data/raw/sentences.csv` with columns: `id`, `class_id`, `sentence`, `lang`, `labels`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or using `uv` (faster):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run Complete Pipeline (Recommended)

To reproduce the full ensemble pipeline with one command:

```bash
./run_full_pipeline.sh
```

This will:
1. Create 80/20 train/test split with 5-fold CV
2. Train 5 folds with ModernBERT-base + LoRA + ASL
3. Tune per-label thresholds on each fold's validation set
4. Run 5-fold ensemble on test set
5. Generate PR curves and evaluation plots

**Expected output:**
- `runs/test_preds.csv` - Final predictions
- `runs/test_preds_metrics.json` - Per-label and aggregate metrics
- `plots/` - PR curves and visualizations

## Step-by-Step Execution

### 1. Create Data Splits

```bash
python -m src.split \
    --input data/raw/sentences.csv \
    --out data/processed/splits.json \
    --test_size 0.2 \
    --folds 5
```

### 2. Train Single Fold

```bash
python -m src.train --config configs/lora_modernbert.yaml --fold 0
```

### 3. Train All Folds

```bash
for fold in {0..4}; do
    python -m src.train --config configs/lora_modernbert.yaml --fold $fold
done
```

Or use the provided script:

```bash
./experiments/run_lora.sh
```

### 4. Tune Thresholds

```bash
python -m src.thresholding \
    --preds runs/fold_0/val_preds.npy \
    --labels runs/fold_0/val_labels.npy \
    --out runs/fold_0/thresholds.json
```

For all folds:

```bash
./experiments/tune_thresholds.sh
```

### 5. Run Inference (Ensemble)

```bash
python -m src.inference \
    --config configs/lora_modernbert.yaml \
    --ckpts "runs/fold_*/best.pt" \
    --ensemble mean \
    --out runs/test_preds.csv \
    --test
```

### 6. Generate Plots

```bash
python -m src.plotting \
    --preds runs/test_preds.csv \
    --labels data/processed/test_labels.npy \
    --out plots/
```

## Run Baselines

### SetFit Baseline

```bash
python -m src.setfit_baseline --config configs/setfit.yaml --fold 0
```

Or:

```bash
./experiments/run_setfit.sh
```

### TF-IDF + Linear SVM Baseline

```bash
python -m src.tfidf_baseline --config configs/tfidf.yaml --fold 0
```

Or:

```bash
./experiments/run_tfidf.sh
```

## Run Tests

```bash
python -m tests.test_asl
python -m tests.test_splits
python -m tests.test_metrics
```

## Configuration

All hyperparameters are in `configs/*.yaml`:

- `configs/lora_modernbert.yaml` - Main model (ModernBERT + LoRA + ASL)
- `configs/setfit.yaml` - SetFit baseline
- `configs/tfidf.yaml` - TF-IDF baseline

Key parameters to tune:
- `train_params.lr` - Learning rate (default: 2e-4)
- `train_params.batch_size` - Batch size (default: 48)
- `peft.r` - LoRA rank (default: 8)
- `loss_params.gamma_neg` - ASL negative focusing (default: 4)

## Expected Performance

With default settings on ~19 labels:

- **ModernBERT + LoRA + ASL**: Micro-F1 > 0.85, Macro-F1 > 0.75
- **SetFit baseline**: Micro-F1 ~ 0.75-0.80
- **TF-IDF baseline**: Micro-F1 ~ 0.65-0.70

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in config (e.g., 32 or 24)
- Increase `grad_accum` to maintain effective batch size
- Reduce `max_len` to 96 or 64

**ModernBERT not available:**
- The code automatically falls back to `microsoft/deberta-v3-base`

**Slow training:**
- Ensure `precision: "bfloat16"` is enabled
- Check `gradient_checkpointing: true`
- Use fewer folds (e.g., 3 instead of 5)

## GPU Memory Usage

With default settings (batch_size=48, max_len=128, bfloat16):
- ModernBERT-base + LoRA: ~18-20GB VRAM
- SetFit (MiniLM): ~8-10GB VRAM

## Next Steps

1. Analyze per-label metrics in `runs/test_preds_metrics.json`
2. Examine PR curves in `plots/`
3. Tune hyperparameters in config files
4. Enable classifier chains: set `chains.enabled: true` in config
5. Experiment with different loss functions: `loss_type: "bce"`
