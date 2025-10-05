# Code Comment Classification

Multi-label classification of code comments using ModernBERT-base with LoRA (PEFT) and Asymmetric Loss (ASL). This project achieves state-of-the-art performance while maintaining simplicity and GPU efficiency (24GB VRAM max).

## Quick Start

### 1. Environment Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or using pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Format

Ensure your dataset is at `data/raw/sentences.csv` with columns:

- `id`: Unique sentence identifier
- `class_id`: Group identifier to avoid leakage across splits (e.g., file/class name)
- `sentence`: Text to classify
- `lang`: Programming language (`JAVA`, `PY`, `PHARO`)
- `labels`: Semicolon- or comma-separated list of label names

### 3. Generate Data Splits

```bash
python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json --test_size 0.2 --folds 5
```

### 4. Train the Model

```bash
python -m src.train --config configs/lora_modernbert.yaml --fold 0
```

### 5. Tune Thresholds

```bash
python -m src.thresholding --preds runs/fold_0/val_preds.npy --labels runs/fold_0/val_labels.npy --out runs/fold_0/thresholds.json
```

### 6. Inference

```bash
python -m src.inference --config configs/lora_modernbert.yaml --ckpts runs/fold_*/best.pt --ensemble mean --out runs/test_preds.csv
```

### 7. Run Baselines

**SetFit Baseline:**

```bash
python -m src.setfit_baseline --config configs/setfit.yaml
```

**TF-IDF + Linear SVM/LogReg Baseline:**

```bash
python -m src.tfidf_baseline --config configs/tfidf.yaml
```

### 8. Generate PR Curves

```bash
python -m src.plotting --preds runs/test_preds.csv --labels data/processed/test_labels.npy --out plots/
```

## Reproducing Final Ensemble

Run these commands to reproduce the complete pipeline:

```bash
python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json --test_size 0.2 --folds 5

for fold in {0..4}; do
  python -m src.train --config configs/lora_modernbert.yaml --fold $fold
done

for fold in {0..4}; do
  python -m src.thresholding --preds runs/fold_${fold}/val_preds.npy --labels runs/fold_${fold}/val_labels.npy --out runs/fold_${fold}/thresholds.json
done

python -m src.inference --config configs/lora_modernbert.yaml --ckpts runs/fold_*/best.pt --ensemble mean --out runs/test_preds.csv

python -m src.plotting --preds runs/test_preds.csv --labels data/processed/test_labels.npy --out plots/
```

This produces:

- Per-label metrics table (CSV): `runs/metrics_per_label.csv`
- Micro/macro-F1 summary: `runs/metrics_summary.json`
- PR curves (PNG): `plots/*.png`
- Final predictions: `runs/test_preds.csv`

## Project Structure

```
code-comment-classification/
├─ README.md
├─ requirements.txt
├─ Makefile
├─ .env.example
├─ .gitignore
├─ configs/
│  ├─ default.yaml
│  ├─ lora_modernbert.yaml
│  ├─ setfit.yaml
│  └─ tfidf.yaml
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ split.py
│  ├─ labels.py
│  ├─ losses.py
│  ├─ models.py
│  ├─ chains.py
│  ├─ train.py
│  ├─ thresholding.py
│  ├─ metrics.py
│  ├─ inference.py
│  ├─ setfit_baseline.py
│  ├─ tfidf_baseline.py
│  ├─ utils.py
│  └─ plotting.py
├─ experiments/
│  ├─ run_lora.sh
│  ├─ run_setfit.sh
│  ├─ run_tfidf.sh
│  └─ tune_thresholds.sh
└─ tests/
   ├─ test_asl.py
   ├─ test_splits.py
   └─ test_metrics.py
```

## Technical Details

- **Model**: ModernBERT-base (149M params) with LoRA (r=8, alpha=16)
- **Loss**: Asymmetric Loss (ASL) with gamma_pos=0, gamma_neg=4, clip=0.05
- **Optimization**: AdamW, lr=2e-4, cosine schedule, 10% warmup
- **Precision**: bfloat16 with gradient checkpointing
- **Splits**: 80/20 holdout + 5-fold iterative stratified group K-fold
- **Evaluation**: Per-label P/R/F1, PR-AUC, micro/macro-F1
- **Memory**: ~20GB VRAM with batch_size=48, max_len=128

## Configuration

All hyperparameters are configurable via YAML files in `configs/`. Key settings:

- `model_name`: Model to use (default: `answerdotai/ModernBERT-base`)
- `peft.enabled`: Enable LoRA (default: true)
- `loss_type`: Loss function (`asl` or `bce`)
- `train_params`: Batch size, learning rate, epochs, etc.
- `chains.enabled`: Enable classifier chains (optional)

## License

MIT