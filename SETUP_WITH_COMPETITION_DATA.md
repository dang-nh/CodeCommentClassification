# Setup Guide with Competition Data

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you prefer conda (recommended):

```bash
conda create -n code-comment python=3.10
conda activate code-comment
pip install -r requirements.txt
```

### 2. Prepare Competition Data

The competition data is already cloned in `data/code-comment-classification/`. Convert it to our format:

```bash
python scripts/prepare_competition_data.py --combine
```

**Output:**
- `data/raw/sentences.csv` - All 6,738 sentences (16 labels, 3 languages)
- `data/raw/java_sentences.csv` - Java only (2,418 sentences, 7 labels)
- `data/raw/python_sentences.csv` - Python only (2,555 sentences, 5 labels)
- `data/raw/pharo_sentences.csv` - Pharo only (1,765 sentences, 7 labels)

### 3. Create Data Splits

```bash
python -m src.split \
    --input data/raw/sentences.csv \
    --out data/processed/splits.json \
    --test_size 0.2 \
    --folds 5
```

This creates group-aware stratified splits with no class_id leakage.

### 4. Train a Model

**Quick test (1 fold, ~30 min on GPU):**

```bash
python -m src.train --config configs/lora_modernbert.yaml --fold 0
```

**Full pipeline (5 folds, ~3-5 hours):**

```bash
./run_full_pipeline.sh
```

## What's Different from Generic Setup

### Dataset
- **Real competition data:** 6,738 sentences from NLBSE'23
- **16 labels** (not 19): Actual categories from Java/Python/Pharo
- **Pre-split:** Competition provides train/test, but we create custom CV folds

### Labels
```
classreferences, collaborators, deprecation, developmentnotes, 
example, expand, intent, keyimplementationpoints, keymessages, 
ownership, parameters, pointer, rational, responsibilities, 
summary, usage
```

### Baseline to Beat
- **Competition baseline:** Random Forest + TF-IDF + NLP features
- **Target:** Micro-F1 > 0.80 (baseline ~0.72-0.74)

## Training Options

### Option 1: All Languages Together (Recommended)

```bash
# Prepare combined data (already done above)
python scripts/prepare_competition_data.py --combine

# Create splits
python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json

# Train
python -m src.train --config configs/lora_modernbert.yaml --fold 0
```

**Advantages:**
- Single model for all languages
- More training data (6,738 sentences)
- Leverages language tokens ([JAVA], [PY], [PHARO])

### Option 2: Per-Language Models

```bash
# Java only
python -m src.split --input data/raw/java_sentences.csv --out data/processed/java_splits.json
python -m src.train --config configs/lora_modernbert.yaml --fold 0

# Python only
python -m src.split --input data/raw/python_sentences.csv --out data/processed/python_splits.json
python -m src.train --config configs/lora_modernbert.yaml --fold 0

# Pharo only
python -m src.split --input data/raw/pharo_sentences.csv --out data/processed/pharo_splits.json
python -m src.train --config configs/lora_modernbert.yaml --fold 0
```

**Advantages:**
- Specialized models per language
- Can tune hyperparameters separately
- Easier to analyze per-language performance

## Evaluation

### Using Competition Test Set

The competition provides a fixed train/test split (partition column). To evaluate on it:

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/raw/sentences.csv')
test_df = df[df['partition'] == 1]  # 1,346 test sentences

# After training, run inference on test_df
# Compare with competition baseline results
```

### Using Our 5-Fold CV

For more robust evaluation:

```bash
# Train all 5 folds
for fold in {0..4}; do
    python -m src.train --config configs/lora_modernbert.yaml --fold $fold
done

# Ensemble predictions
python -m src.inference \
    --config configs/lora_modernbert.yaml \
    --ckpts "runs/fold_*/best.pt" \
    --ensemble mean \
    --out runs/test_preds.csv \
    --test
```

## Expected Results

### Competition Baseline (Random Forest)
- **Java:** Micro-F1 ~0.73, Macro-F1 ~0.71
- **Python:** Micro-F1 ~0.75, Macro-F1 ~0.73
- **Pharo:** Micro-F1 ~0.71, Macro-F1 ~0.69
- **Overall:** Micro-F1 ~0.73, Macro-F1 ~0.71

### Our Model (ModernBERT + LoRA + ASL)
- **Target:** Micro-F1 > 0.80, Macro-F1 > 0.75
- **Expected:** Micro-F1 ~0.82-0.85, Macro-F1 ~0.77-0.80
- **Improvement:** +8-12% over baseline

## Troubleshooting

### ModernBERT Not Available

If ModernBERT download fails, the code automatically falls back to DeBERTa-v3-base. Or manually set:

```yaml
# configs/lora_modernbert.yaml
model_name: "microsoft/deberta-v3-base"
tokenizer_name: "microsoft/deberta-v3-base"
```

### Out of Memory

Reduce batch size:

```yaml
# configs/lora_modernbert.yaml
train_params:
  batch_size: 32  # or 24
  grad_accum: 2   # maintain effective batch size
```

### Slow Training

The full dataset (6,738 sentences) trains in ~30-60 min per fold on RTX 3090/A100. If slower:

1. Check GPU utilization: `nvidia-smi`
2. Ensure bfloat16 is enabled: `precision: "bfloat16"`
3. Reduce epochs for testing: `epochs: 5`

## Verification

After setup, verify everything works:

```bash
# 1. Check data
ls -lh data/raw/sentences.csv
# Should show ~1MB file with 6,738 lines

# 2. Check splits
python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json
# Should create splits.json with 5 folds

# 3. Run tests
python -m tests.test_asl
python -m tests.test_splits
python -m tests.test_metrics
# All should pass

# 4. Quick training test (1 epoch)
python -m src.train --config configs/lora_modernbert.yaml --fold 0
# Should start training without errors
```

## Next Steps

1. **Explore data:** Check `COMPETITION_DATA_GUIDE.md` for dataset details
2. **Run baselines:** Compare TF-IDF and SetFit baselines
3. **Train full model:** Run `./run_full_pipeline.sh`
4. **Analyze results:** Review per-label metrics and PR curves
5. **Tune hyperparameters:** Adjust config based on results

## Resources

- **Competition data guide:** `COMPETITION_DATA_GUIDE.md`
- **Quick start:** `QUICKSTART.md`
- **Reproduction guide:** `REPRODUCE.md`
- **Project summary:** `PROJECT_SUMMARY.md`
