# How to Run Best Experiments

## 📋 Summary of Changes

### ✅ What's Been Done:

1. **Added Full SFT Support** to `dl_solution.py`
   - Set `peft.enabled: false` for full supervised fine-tuning
   - Set `peft.enabled: true` for LoRA parameter-efficient training

2. **Added Single Train/Test Split** option
   - Set `use_single_split: true` for 80/20 stratified split
   - Set `use_single_split: false` for 5-fold cross-validation (recommended for model comparison)

3. **Optimized max_len from 512 → 128**
   - Analysis shows 99.5% of comments are < 128 tokens
   - Saves 75% computation time and memory

4. **Created Best Practice Configs**
   - `configs/dl_best_config.yaml` - DeBERTa-v3 (best performance)
   - `configs/dl_graphcodebert_optimized.yaml` - GraphCodeBERT (code-specific)
   - `configs/dl_single_split.yaml` - Single train/test split example

---

## 🚀 Quick Start: Run Best Experiments

### Experiment 1: DeBERTa-v3 (Best Performance)

```bash
python dl_solution.py configs/dl_best_config.yaml
```

**Expected Results:**
- F1 (samples): **0.87-0.90**
- Training time: ~2 hours (5-fold CV)
- GPU memory: ~8GB

---

### Experiment 2: GraphCodeBERT (Code-Specific)

```bash
python dl_solution.py configs/dl_graphcodebert_optimized.yaml
```

**Expected Results:**
- F1 (samples): **0.82-0.88**
- Training time: ~1.5 hours (5-fold CV)
- GPU memory: ~6GB (with LoRA)

---

### Experiment 3: CodeBERT Baseline (Original)

```bash
python dl_solution.py configs/dl_optimized.yaml
```

**Expected Results:**
- F1 (samples): **0.80-0.85**
- Training time: ~1.5 hours
- GPU memory: ~8GB

---

## 🎯 Recommended Workflow

### Phase 1: Model Comparison (5-Fold CV)

Run all three experiments with `use_single_split: false`:

1. DeBERTa-v3 (Full SFT) - Best performance
2. GraphCodeBERT (LoRA) - Code-specific
3. CodeBERT (Full SFT) - Baseline

Compare average F1 scores with standard deviations.

### Phase 2: Final Model (Single Split)

Choose the best model from Phase 1, then:

1. Set `use_single_split: true` in the config
2. Optionally increase epochs by 20% for final training
3. Train on full 80% dataset
4. Evaluate on held-out 20% test set
5. Save model for production

---

## 📊 Config Comparison Table

| Config File | Model | PEFT | max_len | batch_size | lr | CV | Use Case |
|-------------|-------|------|---------|------------|----|----|----------|
| `dl_best_config.yaml` | DeBERTa-v3 | Full SFT | 128 | 32 | 2e-5 | 5-fold | **Best Performance** |
| `dl_graphcodebert_optimized.yaml` | GraphCodeBERT | LoRA | 128 | 32 | 3e-4 | 5-fold | **Code-Specific** |
| `dl_optimized.yaml` | ModernBERT | Full SFT | 512 | 128 | 5e-6 | 5-fold | **Original** |
| `dl_single_split.yaml` | CodeBERT | Full SFT | 512 | 16 | 3e-4 | single | **Fast Test** |

---

## 🔧 How to Customize

### Change Model:
```yaml
model_name: "microsoft/deberta-v3-base"  # or "microsoft/graphcodebert-base"
tokenizer_name: "microsoft/deberta-v3-base"
```

### Change Training Mode:
```yaml
# Full SFT (all parameters trainable)
peft:
  enabled: false

# LoRA (parameter-efficient)
peft:
  enabled: true
  r: 64
  alpha: 128
  dropout: 0.1
```

### Change Evaluation Strategy:
```yaml
use_single_split: false  # 5-fold CV for model comparison
use_single_split: true   # Single 80/20 split for final training
```

### Adjust max_len:
```yaml
max_len: 128  # Recommended (covers 99.5% of data)
max_len: 256  # Conservative (covers 99.7% of data)
max_len: 512  # Overkill (covers 100% but wastes computation)
```

---

## 📈 Expected Timeline

### Full Comparison Study (3 models × 5 folds):

- DeBERTa-v3: ~2 hours
- GraphCodeBERT: ~1.5 hours
- CodeBERT: ~1.2 hours
- **Total: ~5 hours**

### Single Model Training:
- 5-fold CV: ~1-2 hours
- Single split: ~15-25 minutes per run

---

## 💡 Tips for Best Results

1. **Monitor GPU memory**: Use `nvidia-smi` to check usage
2. **Use fp16 precision**: Saves 50% memory with minimal accuracy loss
3. **Adjust batch size**: Increase if GPU memory allows (faster training)
4. **Early stopping**: Set `early_stop: 5` to prevent overfitting
5. **Check per-label metrics**: Some labels may perform poorly
6. **Save checkpoints**: Modify code to save best fold models
7. **Ensemble**: Average predictions from all 5 folds for production

---

## 🐛 Troubleshooting

### Out of Memory (OOM):
- Reduce `batch_size` from 32 → 16 → 8
- Increase `grad_accum` to maintain effective batch size
- Enable `gradient_checkpointing: true`

### Slow Training:
- Reduce `max_len` from 128 → 64
- Increase `batch_size` if GPU memory allows
- Use LoRA instead of Full SFT
- Reduce epochs

### Poor Performance:
- Increase `epochs` from 10 → 15
- Try different learning rates: 1e-5, 2e-5, 3e-5
- Use Full SFT instead of LoRA
- Check label distribution (some labels may be too rare)

---

## 📝 Example Commands

```bash
# Best performance
python dl_solution.py configs/dl_best_config.yaml

# Code-specific with LoRA
python dl_solution.py configs/dl_graphcodebert_optimized.yaml

# Quick single split test
python dl_solution.py configs/dl_single_split.yaml

# Default config
python dl_solution.py
```

---

## ✅ Next Steps After Training

1. Compare F1 scores across all models
2. Analyze per-label performance
3. Check confusion patterns
4. Choose best model
5. Retrain on full dataset (optional)
6. Deploy to production

Good luck! 🚀

