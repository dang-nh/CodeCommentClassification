# Quick Reference Guide

## 🎯 Your Questions Answered

### ❓ Is max_len=512 enough?

**Answer: YES, but use 128 instead!**

```
Your Data Analysis:
┌─────────────────┬──────────┬────────────┐
│ Percentile      │ Chars    │ Tokens     │
├─────────────────┼──────────┼────────────┤
│ Mean            │ 45       │ 12         │
│ 95th            │ 95       │ 21         │
│ 99th            │ 149      │ 32         │
│ Max             │ 1,321    │ 281        │
└─────────────────┴──────────┴────────────┘

Coverage:
• max_len=128: covers 99.5% of data ✅ RECOMMENDED
• max_len=256: covers 99.7% of data
• max_len=512: covers 100% of data (OVERKILL)
```

---

### ❓ What's the best model?

**Answer: DeBERTa-v3-base (microsoft/deberta-v3-base)**

```
Model Ranking (2024):
┌────┬──────────────────┬─────────┬──────────┬────────────┐
│ #  │ Model            │ F1 Est. │ Speed    │ Best For   │
├────┼──────────────────┼─────────┼──────────┼────────────┤
│ 🥇 │ DeBERTa-v3       │ 0.87-90 │ ⭐⭐⭐   │ Best Score │
│ 🥈 │ GraphCodeBERT    │ 0.82-88 │ ⭐⭐⭐⭐ │ Code Tasks │
│ 🥉 │ CodeBERT         │ 0.80-85 │ ⭐⭐⭐⭐ │ Baseline   │
│    │ ModernBERT       │ 0.78-84 │ ⭐⭐⭐⭐⭐│ Speed      │
└────┴──────────────────┴─────────┴──────────┴────────────┘
```

---

### ❓ What's the best config?

**Answer: Use `configs/dl_best_config.yaml`**

```yaml
# Key Optimizations:
model_name: "microsoft/deberta-v3-base"  # Best model
max_len: 128                              # ← Changed from 512!
batch_size: 32                            # Optimal
lr: 0.00002                               # 2e-5
epochs: 10                                # Enough for convergence
use_single_split: false                   # 5-fold CV for comparison
peft.enabled: false                       # Full SFT for best performance
loss_type: "asl"                          # Handles imbalance
```

---

## 🚀 How to Run

### 1. Best Performance (DeBERTa-v3)
```bash
python dl_solution.py configs/dl_best_config.yaml
```
**Expected:** F1 = 0.87-0.90, ~2 hours, 8GB GPU

### 2. Code-Specific (GraphCodeBERT)
```bash
python dl_solution.py configs/dl_graphcodebert_optimized.yaml
```
**Expected:** F1 = 0.82-0.88, ~1.5 hours, 6GB GPU

### 3. Fast Baseline
```bash
python dl_solution.py configs/dl_optimized.yaml
```
**Expected:** F1 = 0.78-0.84, ~1 hour, 5GB GPU

---

## 📊 Why 5-Fold CV > Single Split?

```
Your Dataset Issues:
├─ Only 6,738 samples (limited)
├─ 16 labels (multi-label)
├─ 22x imbalance (1,712 vs 77 samples)
└─ Rare labels: only 15-25 test samples in single split ❌

5-Fold CV Benefits:
├─ Uses ALL data for training AND validation ✅
├─ Mean ± Std metrics (reliability) ✅
├─ Better for rare labels ✅
└─ Confident model comparison ✅

Recommendation:
1. Use 5-fold CV for model comparison
2. Use single split for final model training
```

---

## 🎓 Web Research Summary

### Latest Best Practices (2024):

**For Multi-Label Classification:**
- ✅ DeBERTa-v3 outperforms BERT/RoBERTa
- ✅ Learning rate: 2e-5 for Full SFT, 3e-4 for LoRA
- ✅ Batch size: 32 is optimal
- ✅ Cosine scheduler with 10% warmup
- ✅ Asymmetric Loss for imbalanced labels
- ✅ Early stopping patience: 5 epochs

**For Code Tasks:**
- ✅ GraphCodeBERT: best code-specific model
- ✅ CodeBERT: solid baseline
- ✅ DeBERTa-v3: often beats code-specific models

**For Limited Data:**
- ✅ Use stratified k-fold CV
- ✅ LoRA for parameter efficiency
- ✅ Early stopping to prevent overfitting
- ✅ Per-label threshold optimization

---

## 📈 Expected Results

```
Experiment              │ F1 Score    │ Time    │ Memory
────────────────────────┼─────────────┼─────────┼────────
DeBERTa-v3 (Full SFT)   │ 0.87-0.90   │ ~2h     │ 8GB
GraphCodeBERT (LoRA)    │ 0.82-0.88   │ ~1.5h   │ 6GB
CodeBERT (Full SFT)     │ 0.80-0.85   │ ~1.2h   │ 6-8GB
ModernBERT              │ 0.78-0.84   │ ~1h     │ 5GB
```

---

## 🔧 Config Options Explained

### Training Mode:
```yaml
peft:
  enabled: false  # Full SFT - all parameters trainable (best performance)
  enabled: true   # LoRA - efficient (good performance, less memory)
```

### Evaluation Strategy:
```yaml
use_single_split: false  # 5-fold CV (for model comparison)
use_single_split: true   # Single 80/20 (for final training)
```

### Max Length:
```yaml
max_len: 128  # Fast, covers 99.5% data ✅ RECOMMENDED
max_len: 256  # Conservative, covers 99.7% data
max_len: 512  # Slow, covers 100% data (overkill)
```

---

## 🎯 Workflow

```
Phase 1: Model Comparison
├─ Run DeBERTa-v3 (5-fold CV)
├─ Run GraphCodeBERT (5-fold CV)
├─ Run CodeBERT (5-fold CV)
└─ Compare: F1_mean ± F1_std

Phase 2: Final Model
├─ Choose best model from Phase 1
├─ Set use_single_split: true
├─ Train on 80% data
├─ Evaluate on 20% test
└─ Save model for production
```

---

## 💡 Pro Tips

1. **Start with `dl_best_config.yaml`** - highest performance
2. **Use 5-fold CV first** - reliable comparison
3. **Monitor GPU** with `nvidia-smi`
4. **Check per-label F1** - overall F1 can be misleading
5. **Use LoRA if OOM** - reduces memory by 50%
6. **Reduce max_len to 64** - for very fast experiments
7. **Ensemble 5 folds** - average predictions for production

---

## 📁 All Configs at a Glance

```
configs/
├─ dl_best_config.yaml               ⭐ DeBERTa-v3, max_len=128, Full SFT
├─ dl_graphcodebert_optimized.yaml   📝 GraphCodeBERT, LoRA
├─ dl_single_split.yaml              ⚡ Single split example
├─ dl_optimized.yaml                 🏠 Original (ModernBERT)
└─ dl_graphcodebert.yaml             📝 Original GraphCodeBERT
```

---

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce batch_size: 32 → 16 → 8
- Use LoRA: set `peft.enabled: true`
- Reduce max_len: 128 → 64

**Training Too Slow?**
- Increase batch_size if memory allows
- Reduce max_len: 128 → 64
- Use single split instead of 5-fold

**Poor Performance?**
- Increase epochs: 10 → 15
- Try Full SFT instead of LoRA
- Check if max_len is too small

---

## ✅ Ready to Run!

```bash
# Quick test (5 minutes)
python dl_solution.py configs/dl_single_split.yaml

# Best model (2 hours)
python dl_solution.py configs/dl_best_config.yaml

# All comparisons (5 hours)
for config in configs/dl_best_config.yaml \
              configs/dl_graphcodebert_optimized.yaml \
              configs/dl_optimized.yaml; do
    python dl_solution.py $config
done
```

**Just specify your conda environment and let's go! 🚀**

